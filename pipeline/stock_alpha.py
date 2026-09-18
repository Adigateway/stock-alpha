"""
stock_alpha_pipeline.py
════════════════════════════════════════════════════════════════════════════════
Full end-to-end stock outperformance prediction pipeline.

Stages (run in order, or toggle via RUN_* flags below):
  1  fetch_financials      → SEC EDGAR quarterly 10-Q data
  2  fetch_prices          → Yahoo Finance daily OHLCV
  3  build_fundamental     → Accounting ratios from raw financials
  4  build_price_features  → Momentum, volatility, returns
  5  build_sentiment       → SEC 8-K filing sentiment (VADER)
  6  build_model_dataset   → Merge + forward-return label
  7  train_model           → HistGradientBoosting + Optuna tuning
  8  backtest_evaluate     → True OOS backtest (train≤2022, test 2023–2025)

Usage:
  python stock_alpha_pipeline.py            # run all stages
  python stock_alpha_pipeline.py --from 3  # resume from stage 3
  python stock_alpha_pipeline.py --only 8  # run only the backtest

════════════════════════════════════════════════════════════════════════════════
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import json
import pickle
from db_config import db_read, db_write, get_engine
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party (auto-install if missing) ─────────────────────────────────────
# _ensure disabled — packages pre-installed

# _ensure("requests", "pandas", "numpy", "yfinance", "scikit-learn",
#         "matplotlib", "seaborn", "optuna", "nltk")

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nltk
import numpy as np
import optuna
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ════════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG  ── edit these to customise the pipeline
# ════════════════════════════════════════════════════════════════════════════════

# DB_PATH removed — using PostgreSQL via db_config.py
USER_AGENT     = "Adigateway adigateway@gmail.com"   # SEC requires a real contact
PRICE_START    = "2010-01-01"
PRICE_END      = pd.Timestamp.today().strftime("%Y-%m-%d")  # always fetch up to today
MIN_YEAR       = 2010
SEC_SLEEP      = 0.15          # seconds between SEC requests (≤10 req/s limit)
MAX_8K_FILINGS = 60            # per ticker
MICROCAP_FLOOR = 300_000_000   # exclude market cap below this
THIN_DATE_MIN  = 80            # drop dates with fewer than this many tickers
TRAIN_END      = "2022-12-31"  # hard wall — model never sees beyond this
TEST_START     = "2023-01-01"  # OOS test window start
HORIZON_M      = 6             # forward-return prediction horizon (months)
FILING_LAG_DAYS = 45           # days after period-end before a 10-Q is public
                               # SEC "end" is the quarter END, not the filing
                               # date. Without this lag the model reads numbers
                               # before they existed publicly.
N_OPTUNA       = 50            # Optuna trials (50 gives better convergence)
N_CV_FOLDS     = 8             # outer time-series CV folds
HALF_LIFE_M    = 36            # time-decay half-life (months) for sample weights

HEADERS = {"User-Agent": USER_AGENT}

# SEC XBRL tags → our column names
# SEC XBRL tags -> our column names.
# Order matters: earlier tags take priority per-date, later tags fill gaps.
# ALL tags are tried (no break) — filers switch tags over time, most notably
# from "Revenues" to the ASC 606 tag around 2018.
FIELD_MAP = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606, post-2018
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",                                             # legacy, pre-2018
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "gross_profit":        ["GrossProfit"],
    "cost_of_revenue": [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    ],
    "operating_income":    ["OperatingIncomeLoss"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "total_assets":        ["Assets"],
    "current_assets":      ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "operating_cashflow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",          # lives in the dei namespace
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
}

# ── Period-length rules for duration concepts ────────────────────────────────
# Balance-sheet items are instant (no start date) and always accepted.
#
# Income-statement concepts are tagged per-quarter by most filers, so a
# 60-120 day window keeps the series consistent and stops a 10-K's full-year
# revenue from making Q4 look like a 4x jump.
#
# Cash-flow statements are the exception: in 10-Qs they are CUMULATIVE
# year-to-date, so Q2 spans ~180 days and Q3 ~270. Filtering those out leaves
# only Q1 and destroys coverage. We accept the wider window and annualise.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 120

PERIOD_RULES = {
    "operating_cashflow": (60, 400),   # YTD cumulative in 10-Qs
}

# Fields whose value is scaled to a 365-day equivalent: val * 365 / days.
# Makes YTD and annual figures directly comparable.
ANNUALIZE_FIELDS = {"operating_cashflow"}

# 8-K item → descriptive text for VADER scoring
ITEM_DESCRIPTIONS = {
    "1.01": "entry into material definitive agreement partnership deal",
    "1.02": "termination of material definitive agreement loss contract ended",
    "1.03": "bankruptcy insolvency financial distress default",
    "2.01": "completion of acquisition merger growth expansion strategic deal",
    "2.02": "results of operations earnings revenue financial performance quarterly results",
    "2.03": "creation of direct financial obligation debt borrowing loan",
    "2.04": "triggering events accelerate obligation default impairment loss",
    "2.05": "costs associated with exit disposal restructuring layoffs charges",
    "2.06": "material impairment write-down asset loss charge",
    "3.01": "notice delisting listing transfer exchange concern",
    "4.02": "non-reliance financial statements restatement error correction",
    "5.02": "departure directors officers executive leadership change CEO CFO",
    "7.01": "regulation FD disclosure guidance outlook forecast",
    "8.01": "other material events significant news announcement",
}

# Base ML features (engineered + relative features added dynamically)
BASE_FEATURES = [
    # Ablation-validated features only (see feature_ablation.json)
    # All fundamentals, sentiment, and topic features failed empirical testing
    "mom_3m", "mom_6m", "mom_6m_rank", "vol_6m",
]

# Confidence threshold for HIGH-conviction label in prediction tables
HIGH_CONF_THRESHOLD = 0.60   # lowered from 0.65 — surfaces more strong calls

# ════════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def banner(title: str, stage: int = None):
    tag = f"[Stage {stage}]  " if stage else ""
    line = "═" * 72
    print(f"\n{line}")
    print(f"  {tag}{title}")
    print(f"{line}\n")

# db_read and db_write imported from db_config.py

def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))

def sector_median_impute(df: pd.DataFrame, col: str) -> pd.Series:
    if "sector" in df.columns:
        return df[col].fillna(df.groupby("sector")[col].transform("median"))
    return df[col].fillna(df[col].median())

# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FETCH FINANCIALS (SEC EDGAR)
# ════════════════════════════════════════════════════════════════════════════════

def fetch_financials():
    banner("Fetch quarterly 10-Q financials from SEC EDGAR", stage=1)

    # ── Ticker universe ──────────────────────────────────────────────────────
    # Read from ticker_sectors in the DB. This is the stable universe every
    # other stage joins against. Scraping Wikipedia is a fallback only — its
    # page structure changes and would silently alter the universe mid-project.
    try:
        _ts = db_read("SELECT DISTINCT ticker FROM ticker_sectors ORDER BY ticker")
        tickers = _ts["ticker"].dropna().astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t]
        if len(tickers) < 50:
            raise ValueError(f"only {len(tickers)} tickers in ticker_sectors")
        print(f"  Tickers from ticker_sectors : {len(tickers)}")
    except Exception as _e:
        print(f"  ticker_sectors unavailable ({_e}) — falling back to Wikipedia")
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        _tables = pd.read_html(resp.text)
        _tbl = next((t for t in _tables if "Symbol" in t.columns), None)
        if _tbl is None:
            raise RuntimeError(
                "No table with a 'Symbol' column found on the Wikipedia page. "
                "Populate ticker_sectors instead.")
        tickers = _tbl["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        print(f"  S&P 500 tickers from Wikipedia : {len(tickers)}")

    # CIK map
    cik_data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS).json()
    ticker_to_cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in cik_data.values()}
    ticker_cik_map = {t: ticker_to_cik[t] for t in tickers if t in ticker_to_cik}
    print(f"  CIK-matched tickers   : {len(ticker_cik_map)}")

    def _period_days(item: dict):
        """
        Length of the period this fact covers, in days.

        Returns None for instant concepts (balance-sheet items have no
        "start"), or -1 if the dates cannot be parsed.
        """
        start = item.get("start")
        end   = item.get("end")
        if not start or not end:
            return None
        try:
            return (pd.Timestamp(end) - pd.Timestamp(start)).days
        except Exception:
            return -1

    def extract_quarterly(facts: dict, ticker: str) -> list:
        """
        Pull quarterly fundamentals from a SEC companyfacts blob.

        Reads BOTH the us-gaap and dei namespaces, tries EVERY tag in each
        field's list (earlier tags win per-date, later ones fill gaps), and
        accepts 10-K filings so fiscal Q4 is not silently dropped.
        """
        all_facts = facts.get("facts", {})
        us_gaap   = all_facts.get("us-gaap", {})
        dei       = all_facts.get("dei", {})

        records: dict = {}

        for field, tags in FIELD_MAP.items():
            for tag in tags:
                # A tag may live in either namespace — dei holds entity-level
                # facts like EntityCommonStockSharesOutstanding.
                node = us_gaap.get(tag) or dei.get(tag)
                if node is None:
                    continue

                units = node.get("units", {})
                if not units:
                    continue
                data = (units.get("USD")
                        or units.get("shares")
                        or units.get("USD/shares")
                        or next(iter(units.values()), []))

                lo, hi = PERIOD_RULES.get(field, (QUARTER_MIN_DAYS, QUARTER_MAX_DAYS))

                for item in data:
                    if item.get("form") not in ("10-Q", "10-K"):
                        continue

                    days = _period_days(item)
                    if days is not None:            # duration concept
                        if days < 0 or not (lo <= days <= hi):
                            continue

                    date = item.get("end")
                    if not date:
                        continue
                    try:
                        if int(str(date)[:4]) < MIN_YEAR:
                            continue
                    except (ValueError, TypeError):
                        continue

                    val = item.get("val")
                    if val is None:
                        continue

                    # Scale YTD / annual figures to a 365-day equivalent so
                    # they are comparable across filings and fiscal calendars.
                    if field in ANNUALIZE_FIELDS and days and days > 0:
                        val = val * 365.0 / days

                    rec = records.setdefault(
                        date, {"ticker": ticker, "report_date": date}
                    )

                    if rec.get(field) is None:
                        rec[field] = val
                        rec[f"_{field}_days"] = days
                    elif field in ANNUALIZE_FIELDS and days:
                        # Prefer the longest period available for this date —
                        # a YTD figure is less seasonally noisy than a single
                        # quarter once annualised.
                        prev_days = rec.get(f"_{field}_days") or 0
                        if days > prev_days:
                            rec[field] = val
                            rec[f"_{field}_days"] = days
                # NO break — continue to the next tag to fill remaining dates

        return list(records.values())

    all_records = []
    for i, (ticker, cik) in enumerate(ticker_cik_map.items(), 1):
        print(f"  [{i:>3}/{len(ticker_cik_map)}] {ticker:<6}", end=" ", flush=True)
        try:
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                             headers=HEADERS, timeout=15)
            r.raise_for_status()
            qtrs = extract_quarterly(r.json(), ticker)
            all_records.extend(qtrs)
            print(f"→ {len(qtrs)} quarters")
        except Exception as e:
            print(f"→ FAILED ({e})")
        time.sleep(SEC_SLEEP)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError("No data — check SEC connectivity.")

    # Derive gross_profit where the GrossProfit tag is absent but revenue and
    # cost_of_revenue are both present. Filers tag GrossProfit inconsistently.
    if {"revenue", "cost_of_revenue"} <= set(df.columns):
        if "gross_profit" not in df.columns:
            df["gross_profit"] = np.nan
        _derivable = (
            df["gross_profit"].isna()
            & df["revenue"].notna()
            & df["cost_of_revenue"].notna()
        )
        df.loc[_derivable, "gross_profit"] = (
            df.loc[_derivable, "revenue"] - df.loc[_derivable, "cost_of_revenue"]
        )
        print(f"  gross_profit derived for {int(_derivable.sum()):,} rows "
              f"(revenue - cost_of_revenue)")

    # Internal period-tracking columns are not part of the schema
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")],
                 errors="ignore")

    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["ticker", "report_date"]).dropna(subset=["revenue"])

    # ── Coverage report — makes extraction failures visible ──────────────────
    per_ticker = df.groupby("ticker").agg(
        quarters=("report_date", "count"),
        newest=("report_date", "max"),
    )
    thin  = per_ticker[per_ticker["quarters"] < 20]
    stale = per_ticker[per_ticker["newest"] < (pd.Timestamp.today() - pd.Timedelta(days=200))]

    print(f"\n  ── Coverage ──────────────────────────────────────────────")
    for _fld in ("revenue", "operating_cashflow", "total_equity", "gross_profit"):
        if _fld in df.columns:
            print(f"    {_fld:<20} present on {df[_fld].notna().mean():.0%} of rows")
    print(f"    Median quarters per ticker : {per_ticker['quarters'].median():.0f}")
    print(f"    Tickers with <20 quarters  : {len(thin)} of {len(per_ticker)}")
    print(f"    Tickers stale >200 days    : {len(stale)} of {len(per_ticker)}")
    if len(thin):
        worst = thin.sort_values("quarters").head(10)
        print(f"    Thinnest: " + ", ".join(
            f"{t}({int(r.quarters)})" for t, r in worst.iterrows()))
    for _chk in ("AAPL", "MSFT", "NVDA", "JPM", "HD"):
        if _chk in per_ticker.index:
            _r = per_ticker.loc[_chk]
            print(f"    {_chk:<6} {int(_r['quarters']):>3} quarters, "
                  f"newest {_r['newest'].date()}")
        else:
            print(f"    {_chk:<6} MISSING")

    db_write(df, "financials")
    print(f"\n  ✓ financials  →  {len(df):,} rows  |  {df['ticker'].nunique()} tickers")
    print(f"    Date range: {df['report_date'].min().date()} → {df['report_date'].max().date()}")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FETCH PRICES (Yahoo Finance)
# ════════════════════════════════════════════════════════════════════════════════

def fetch_prices():
    banner("Fetch daily OHLCV prices from Yahoo Finance", stage=2)

    tickers = db_read("SELECT DISTINCT ticker FROM financials ORDER BY ticker")["ticker"].tolist()
    print(f"  Downloading {len(tickers)} tickers  |  {PRICE_START} → {PRICE_END}\n")

    raw = yf.download(
        tickers, start=PRICE_START, end=PRICE_END,
        progress=True, group_by="ticker", threads=True, auto_adjust=True,
    )

    frames = []
    for ticker in tickers:
        try:
            df = raw[ticker][["Close", "Volume"]].copy() if len(tickers) > 1 else raw[["Close", "Volume"]].copy()
            df["ticker"] = ticker
            df["date"]   = df.index
            frames.append(df.reset_index(drop=True))
        except Exception as e:
            print(f"  Warning: skipped {ticker} — {e}")

    df_all = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["Close"])
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
        .sort_values(["ticker", "date"])
    )

    db_write(df_all, "prices")
    print(f"\n  ✓ prices  →  {len(df_all):,} rows  |  {df_all['ticker'].nunique()} tickers")
    print(f"    Date range: {df_all['date'].min().date()} → {df_all['date'].max().date()}")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BUILD FUNDAMENTAL FEATURES
# ════════════════════════════════════════════════════════════════════════════════

def build_fundamental_features():
    banner("Build fundamental (accounting) ratio features", stage=3)

    df = db_read("SELECT * FROM financials")
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["ticker", "report_date"])

    assert df["total_assets"].median() > 1_000, \
        "total_assets looks log-transformed — check Stage 1."

    # Profitability
    df["roa"]              = df["net_income"]       / df["total_assets"]
    df["roe"]              = df["net_income"]       / df["total_equity"]
    df["operating_margin"] = df["operating_income"] / df["revenue"]

    # Leverage
    df["debt_to_equity"]   = df["total_debt"] / df["total_equity"]
    df["debt_to_assets"]   = df["total_debt"] / df["total_assets"]

    # Liquidity
    df["current_ratio"]    = df["current_assets"] / df["current_liabilities"]

    # Growth (YoY, 4 quarters)
    df["revenue_yoy"]      = df.groupby("ticker")["revenue"].pct_change(4)
    df["income_yoy"]       = df.groupby("ticker")["net_income"].pct_change(4)
    df["asset_growth"]     = df.groupby("ticker")["total_assets"].pct_change(4)

    # Quality
    df["accrual_ratio"]    = (df["net_income"] - df["operating_cashflow"]) / df["total_assets"]

    FACTOR_COLS = [
        "roa", "roe", "operating_margin",
        "debt_to_equity", "debt_to_assets", "current_ratio",
        "revenue_yoy", "income_yoy", "asset_growth", "accrual_ratio",
    ]

    # Clip & impute
    for col in FACTOR_COLS:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
        df[col] = sector_median_impute(df, col)

    df = df.replace([np.inf, -np.inf], np.nan)

    db_write(df, "fundamental_factors")
    print(f"  ✓ fundamental_factors  →  {len(df):,} rows")
    null_pct = df[FACTOR_COLS].isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if not high_null.empty:
        print("\n  Warning — high null rates:")
        print(high_null.to_string())


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 4 — BUILD PRICE FEATURES
# ════════════════════════════════════════════════════════════════════════════════

def build_price_features():
    banner("Build price momentum & volatility features", stage=4)

    prices = db_read("SELECT * FROM prices")
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    prices = prices.sort_values(["ticker", "date"])

    # 1. Compute raw features
    prices["return"]  = prices.groupby("ticker")["Close"].pct_change()
    prices["mom_3m"]  = prices.groupby("ticker")["Close"].pct_change(63)
    prices["mom_6m"]  = prices.groupby("ticker")["Close"].pct_change(126)
    prices["mom_12m"] = prices.groupby("ticker")["Close"].pct_change(252)

    prices["vol_6m"]  = (
        prices.groupby("ticker")["return"]
        .transform(lambda x: x.rolling(126, min_periods=20).std())
    )
    prices["mom_12_2"] = (
        prices.groupby("ticker")["Close"]
        .transform(lambda x: x.shift(21) / x.shift(252) - 1)
    )

    # 2. Fill NAs BEFORE cross-sectional operations
    for col in ["return", "mom_3m", "mom_6m", "mom_12m", "mom_12_2"]:
        prices[col] = prices[col].fillna(0)
    prices["vol_6m"] = prices["vol_6m"].fillna(
        prices.groupby("ticker")["vol_6m"].transform("median")
    )

    # 3. Cross-sectional ranking (per date across universe)
    prices["mom_12_2_rank"] = prices.groupby("date")["mom_12_2"].rank(pct=True)
    prices["mom_6m_rank"]   = prices.groupby("date")["mom_6m"].rank(pct=True)

    # 4. Cross-sectional Winsorization (per date to eliminate lookahead bias)
    for col in ["return", "mom_3m", "mom_6m", "mom_12m", "mom_12_2", "vol_6m"]:
        prices[col] = prices.groupby("date")[col].transform(winsorize)

    # Log-transform vol
    prices["vol_6m"] = np.log(prices["vol_6m"].clip(lower=1e-8))

    db_write(prices, "price_factors")
    print(f"  ✓ price_factors  →  {len(prices):,} rows  |  {prices['ticker'].nunique()} tickers")
# ════════════════════════════════════════════════════════════════════════════════
# STAGE 5 — BUILD SENTIMENT
# Real 8-K document text  →  TF-IDF  →  TruncatedSVD topics  →  VADER fallback
#
# Approach (inspired by hatemr/NLP-for-8K-documents):
#   1. Fetch 8-K filing list + accession numbers from SEC EDGAR
#   2. Download actual document text for each filing (cached in SQLite)
#   3. Clean & normalise text (strip HTML, boilerplate, legal noise)
#   4. VADER score on real text (not item-description proxies)
#   5. TF-IDF vectorise all filing texts → TruncatedSVD → N_TOPICS topic weights
#   6. Aggregate all signals quarterly per ticker
#   7. Topic weights become extra model features (topic_0 … topic_N)
#
# Fallback: if EDGAR text fetch fails, VADER scores the item-description text
#           (same as old pipeline) so we never lose a filing entirely.
# ════════════════════════════════════════════════════════════════════════════════

# ── NLP config ───────────────────────────────────────────────────────────────
N_TOPICS          = 15    # TruncatedSVD components — matches repo findings
TFIDF_MAX_FEAT    = 8000  # vocabulary cap
TFIDF_MIN_DF      = 3     # ignore terms appearing in fewer than N docs
TFIDF_MAX_DF      = 0.85  # ignore terms appearing in >85% of docs (boilerplate)
MAX_TEXT_CHARS    = 6000  # chars to read per filing (first ~1000 words is enough)

def build_sentiment():
    banner("Build sentiment — TF-IDF + SVD on real 8-K text (free, no API)", stage=5)

#     _ensure("scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    import re, html

    # ── Domain-adapted VADER (kept as fallback & VADER score on real text) ──
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update({
        "beat": 3.0,  "beats": 3.0,  "exceeded": 2.5,  "record": 2.0,
        "surpassed": 2.5, "outperformed": 2.5, "raised": 2.0, "upgrade": 2.5,
        "upgraded": 2.5, "profitable": 2.0, "profitability": 2.0, "growth": 1.5,
        "buyback": 1.5, "dividend": 1.5, "bullish": 3.0, "upside": 2.0,
        "miss": -3.0, "missed": -3.0, "below": -1.5, "loss": -2.0,
        "losses": -2.0, "declined": -2.0, "lowered": -2.0, "downgrade": -2.5,
        "downgraded": -2.5, "disappointing": -2.5, "weak": -2.0, "headwinds": -2.0,
        "uncertainty": -1.5, "lawsuit": -2.0, "recall": -2.5, "bankruptcy": -3.5,
        "fraud": -3.5, "investigation": -2.5, "layoffs": -2.0, "restructuring": -1.5,
        "bearish": -3.0, "shortfall": -2.5, "restatement": -3.0, "default": -3.0,
    })

    def items_to_text(items_str: str) -> str:
        """Fallback: convert item numbers to descriptive text (old VADER approach)."""
        parts = [ITEM_DESCRIPTIONS.get(i.strip(), "") for i in str(items_str).split(",")]
        return " ".join(p for p in parts if p) or "company filing announcement"

    def vader_score(text: str) -> float:
        if not text or len(text.strip()) < 5:
            return 0.0
        return sia.polarity_scores(text)["compound"]

    # ── Text cleaning ─────────────────────────────────────────────────────────
    # Financial boilerplate patterns to strip before TF-IDF
    _BOILERPLATE = re.compile(
        r"pursuant to|in accordance with|section \d+|exhibit \d+|"
        r"form \d+[-\w]*|securities exchange act|herein by reference|"
        r"registrant|signatur|officer's certificate|underwriting agreement|"
        r"forward.looking statement|safe harbor|risk factor",
        re.IGNORECASE
    )
    _HTML_TAG    = re.compile(r"<[^>]+>")
    _WHITESPACE  = re.compile(r"\s+")
    _NON_ALPHA   = re.compile(r"[^a-zA-Z\s]")

    def clean_text(raw: str) -> str:
        """Strip HTML, boilerplate, digits, and normalise whitespace."""
        text = html.unescape(raw)
        text = _HTML_TAG.sub(" ", text)         # remove HTML tags
        text = _BOILERPLATE.sub(" ", text)       # strip legal boilerplate
        text = _NON_ALPHA.sub(" ", text)         # keep only letters
        text = text.lower()
        text = _WHITESPACE.sub(" ", text).strip()
        return text[:MAX_TEXT_CHARS]             # cap length

    # ── EDGAR text fetcher ────────────────────────────────────────────────────
    def fetch_filing_text(cik_int: int, accession: str, primary_doc: str) -> str | None:
        """
        Fetch actual 8-K document text from EDGAR.
        cik_int    : CIK as integer (no leading zeros)
        accession  : accession number with dashes e.g. '0000320193-17-000070'
        primary_doc: filename from filing index e.g. '0000320193-17-000070.htm'
        Returns cleaned text or None on failure.
        """
        acc_nodash = accession.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{cik_int}/{acc_nodash}/{primary_doc}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return clean_text(r.text)
        except Exception:
            # Try the index to find alternate primary doc name
            try:
                idx_url = (f"https://www.sec.gov/Archives/edgar/data/"
                           f"{cik_int}/{acc_nodash}/{acc_nodash}-index.json")
                idx = requests.get(idx_url, headers=HEADERS, timeout=10).json()
                docs = idx.get("documents", [])
                for doc in docs:
                    if doc.get("type") in ("8-K", "8-K/A") or doc.get("sequence") == "1":
                        alt_url = (f"https://www.sec.gov/Archives/edgar/data/"
                                   f"{cik_int}/{acc_nodash}/{doc['name']}")
                        r2 = requests.get(alt_url, headers=HEADERS, timeout=12)
                        r2.raise_for_status()
                        return clean_text(r2.text)
            except Exception:
                pass
        return None

    # ── Load ticker → CIK map ─────────────────────────────────────────────────
    tickers = db_read("SELECT DISTINCT ticker FROM financials ORDER BY ticker")["ticker"].tolist()
    cik_data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS).json()
    ticker_to_cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in cik_data.values()}
    ticker_cik_map = {t: ticker_to_cik[t] for t in tickers if t in ticker_to_cik}
    print(f"  Tickers to process : {len(ticker_cik_map)}")

    # ── Load text cache (avoid re-fetching already-downloaded filings) ────────
    # Normalise accession format on load: strip whitespace, ensure consistent form
    try:
        cache_df = db_read("SELECT accession, text FROM sec_8k_text_cache")
        # Deduplicate on accession (keep last in case of prior duplicate writes)
        cache_df = cache_df.drop_duplicates(subset="accession", keep="last")
        # Normalise: strip whitespace (guards against any format drift)
        cache_df["accession"] = cache_df["accession"].str.strip()
        text_cache = dict(zip(cache_df["accession"], cache_df["text"]))
        print(f"  Text cache loaded  : {len(text_cache):,} filings already cached")
        print(f"  (Run will only fetch filings NOT already in cache)")
    except Exception:
        text_cache = {}
        print("  Text cache         : empty (first run — all texts will be fetched)")
        print("  (Subsequent runs will be fast — texts are cached permanently)")

    # ── Phase 1: collect filing metadata + fetch texts ────────────────────────
    print("\n── Phase 1: fetching 8-K filings & document text ───────────────────")

    all_filings   = []   # list of dicts: ticker, date, items, accession, text, vader_score
    new_cache_rows = []  # rows to add to text cache

    for i, (ticker, cik) in enumerate(ticker_cik_map.items(), 1):
        print(f"  [{i:>3}/{len(ticker_cik_map)}] {ticker:<6}", end=" ", flush=True)
        try:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                             headers=HEADERS, timeout=12)
            r.raise_for_status()
            sub   = r.json()
            recent = sub.get("filings", {}).get("recent", {})

            forms    = recent.get("form", [])
            dates    = recent.get("filingDate", [])
            items_l  = recent.get("items", [])
            accs     = recent.get("accessionNumber", [])
            pdocs    = recent.get("primaryDocument", [])
            cik_int  = int(cik)

            filing_count = 0
            fetched_text = 0
            for j, (form, date) in enumerate(zip(forms, dates)):
                if form != "8-K":
                    continue
                try:
                    if int(date[:4]) < MIN_YEAR:
                        continue
                except ValueError:
                    continue

                acc      = accs[j]  if j < len(accs)   else ""
                pdoc     = pdocs[j] if j < len(pdocs)  else ""
                items    = items_l[j] if j < len(items_l) else ""

                # Try to get real text (cached or fetch)
                doc_text = None
                if acc in text_cache:
                    doc_text = text_cache[acc]
                elif acc and pdoc:
                    time.sleep(SEC_SLEEP)
                    doc_text = fetch_filing_text(cik_int, acc, pdoc)
                    if doc_text:
                        text_cache[acc] = doc_text
                        new_cache_rows.append({"accession": acc, "text": doc_text})
                        fetched_text += 1

                # Score: real text if available, else fallback to item description
                score_text = doc_text if doc_text else items_to_text(items)
                score      = vader_score(score_text)
                label      = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
                source     = "text" if doc_text else "items"

                all_filings.append({
                    "ticker"         : ticker,
                    "date"           : date,
                    "items"          : items,
                    "accession"      : acc,
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "score_source"   : source,
                })
                filing_count += 1
                if filing_count >= MAX_8K_FILINGS:
                    break

            time.sleep(SEC_SLEEP)

            # ── Write new cache rows to DB immediately after each ticker ──────
            # This means a crash/interrupt never loses more than 1 ticker of work
            if new_cache_rows:
                cache_new_df = pd.DataFrame(new_cache_rows)
                db_write(cache_new_df, "sec_8k_text_cache", if_exists="append")
                new_cache_rows = []   # reset for next ticker

            print(f"→ {filing_count:>2} filings  "
                  f"({fetched_text} new  |  cached: {filing_count - fetched_text})")

        except Exception as e:
            print(f"→ FAILED ({e})")
            time.sleep(SEC_SLEEP)

    if not all_filings:
        raise RuntimeError("No filings collected — check SEC connectivity.")

    df = pd.DataFrame(all_filings)
    df["date"] = pd.to_datetime(df["date"])

    text_coverage = (df["score_source"] == "text").mean()
    print(f"  Filing text coverage : {text_coverage:.1%} of filings used real document text")
    print(f"  VADER fallback       : {1-text_coverage:.1%} of filings used item-description proxy")

    # ── Phase 2: TF-IDF + TruncatedSVD on all texts that have real content ───
    print("\n── Phase 2: TF-IDF vectorisation + TruncatedSVD topic modelling ─────")

    real_text_mask = df["score_source"] == "text"
    n_real = real_text_mask.sum()
    print(f"  Fitting TF-IDF on {n_real:,} real filing texts")
    print(f"  Vocab cap: {TFIDF_MAX_FEAT:,} terms  |  Topics: {N_TOPICS}")

    topic_cols = [f"topic_{k}" for k in range(N_TOPICS)]

    # Initialise topic columns to 0 for all rows
    for col in topic_cols:
        df[col] = 0.0

    if n_real >= max(N_TOPICS * 3, 50):  # need enough docs to fit SVD
        real_texts = df.loc[real_text_mask, "accession"].map(text_cache).fillna("").tolist()

        # TF-IDF — min_df/max_df filters boilerplate terms automatically
        tfidf = TfidfVectorizer(
            max_features = TFIDF_MAX_FEAT,
            min_df       = TFIDF_MIN_DF,
            max_df       = TFIDF_MAX_DF,
            ngram_range  = (1, 2),      # unigrams + bigrams for better phrases
            sublinear_tf = True,        # log(tf) dampens very common terms
            strip_accents = "unicode",
            stop_words   = "english",
        )
        X_tfidf = tfidf.fit_transform(real_texts)
        print(f"  TF-IDF matrix       : {X_tfidf.shape[0]:,} docs × {X_tfidf.shape[1]:,} terms")

        # TruncatedSVD (LSA — latent semantic analysis)
        svd = TruncatedSVD(n_components=N_TOPICS, random_state=42, n_iter=7)
        X_svd = svd.fit_transform(X_tfidf)

        # L2-normalise so topic weights are comparable across filings
        X_norm = Normalizer(copy=False).fit_transform(X_svd)

        variance_explained = svd.explained_variance_ratio_.sum()
        print(f"  SVD variance explained: {variance_explained:.1%} by {N_TOPICS} topics")

        # Assign topic weights back to the real-text rows
        df.loc[real_text_mask, topic_cols] = X_norm

        # Show top words per topic for interpretability
        print("\n  Top terms per topic:")
        terms = tfidf.get_feature_names_out()
        for k in range(min(5, N_TOPICS)):   # show first 5 topics
            top_idx = svd.components_[k].argsort()[-8:][::-1]
            top_words = " | ".join(terms[top_idx])
            print(f"    Topic {k}: {top_words}")
    else:
        print(f"  Warning: only {n_real} real texts — skipping SVD (need ≥{max(N_TOPICS*3,50)})")
        print("  Topic features will all be 0. Run Stage 5 again after cache warms up.")

    # ── Phase 3: Quarterly aggregation ───────────────────────────────────────
    print("\n── Phase 3: aggregating to quarterly features ───────────────────────")

    df["quarter"] = df["date"].dt.to_period("Q").apply(lambda r: r.start_time)

    # Aggregate sentiment scores
    sent_agg = (
        df.groupby(["ticker", "quarter"])
        .agg(
            sentiment_avg     = ("sentiment_score", "mean"),
            sentiment_bullish = ("sentiment_label", lambda x: (x == "positive").mean()),
            sentiment_bearish = ("sentiment_label", lambda x: (x == "negative").mean()),
            sentiment_vol     = ("sentiment_score", "std"),
            sentiment_count   = ("sentiment_score", "count"),
            text_coverage     = ("score_source",    lambda x: (x == "text").mean()),
        )
        .reset_index()
        .rename(columns={"quarter": "date"})
    )

    # Aggregate topic weights (mean per quarter)
    topic_agg = (
        df.groupby(["ticker", "quarter"])[topic_cols]
        .mean()
        .reset_index()
        .rename(columns={"quarter": "date"})
    )

    quarterly = sent_agg.merge(topic_agg, on=["ticker", "date"], how="left")

    # Clean up
    quarterly["sentiment_vol"]   = quarterly["sentiment_vol"].fillna(0)
    quarterly["sentiment_count"] = np.log1p(quarterly["sentiment_count"])
    for col in ["sentiment_avg", "sentiment_vol"]:
        quarterly[col] = winsorize(quarterly[col])
    for col in topic_cols:
        quarterly[col] = quarterly[col].fillna(0.0)

    # ── Save ─────────────────────────────────────────────────────────────────
    raw_cols = ["ticker", "date", "items", "accession",
                "sentiment_label", "sentiment_score", "score_source"]
    db_write(df[[c for c in raw_cols if c in df.columns]], "sec_sentiment_raw")
    db_write(quarterly, "sentiment_features")

    print(f"\n  ✓ sec_sentiment_raw    →  {len(df):,} rows")
    print(f"  ✓ sentiment_features   →  {len(quarterly):,} rows  |  {quarterly['ticker'].nunique()} tickers")
    print(f"  ✓ sec_8k_text_cache    →  {len(text_cache):,} total cached filings")
    print(f"  ✓ Topic columns added  :  {topic_cols}")
    print(f"\n  NOTE: Add topic_0…topic_{N_TOPICS-1} to BASE_FEATURES in config")
    print(f"        to include them in the ML model (already done if using")
    print(f"        the current pipeline version).")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 6 — BUILD MODEL DATASET
# ════════════════════════════════════════════════════════════════════════════════

def build_model_dataset():
    banner("Merge all features → clean model dataset", stage=6)

    fund  = db_read("SELECT * FROM fundamental_factors")
    price = db_read("SELECT * FROM price_factors")
    fund["report_date"] = pd.to_datetime(fund["report_date"])
    price["date"]       = pd.to_datetime(price["date"])

    # Monthly resample (last close of each month)
    monthly = (
        price.sort_values(["ticker", "date"])
        .set_index(["ticker", "date"])
        .groupby(level=0)
        .resample("ME", level=1)
        .last()
        .reset_index()
    )

    # Forward 6-month return label
    monthly["fwd_6m_return"] = (
        monthly.groupby("ticker")["Close"].shift(-HORIZON_M) / monthly["Close"] - 1
    )

    # ── Forward-fill shares_outstanding within each ticker ───────────────────
    # SEC XBRL omits CommonStockSharesOutstanding on roughly half of filings.
    # Without this, market_cap is NaN, and the MICROCAP_FLOOR filter below
    # silently deletes the company from the universe entirely. Share counts
    # move slowly, so carrying the last known value forward is far more
    # accurate than dropping the row.
    fund = fund.sort_values(["ticker", "report_date"])
    if "shares_outstanding" in fund.columns:
        _na_before = int(fund["shares_outstanding"].isna().sum())
        fund["shares_outstanding"] = fund.groupby("ticker")["shares_outstanding"].ffill()
        # bfill only fills leading gaps where a ticker never reported shares
        # in its earliest filings. Share counts are stable enough that this is
        # a minor approximation, not meaningful look-ahead.
        fund["shares_outstanding"] = fund.groupby("ticker")["shares_outstanding"].bfill()
        _na_after = int(fund["shares_outstanding"].isna().sum())
        print(f"  shares_outstanding : {_na_before:,} missing -> {_na_after:,} "
              f"({_na_before - _na_after:,} recovered by ffill/bfill)")

    # ── Filing lag ───────────────────────────────────────────────────────────
    # report_date is the SEC "end" field = the quarter's PERIOD END, not the
    # date the 10-Q became public. Merging on period-end means the model sees
    # fundamentals ~45 days before they were actually available.
    fund["available_date"] = fund["report_date"] + pd.Timedelta(days=FILING_LAG_DAYS)
    print(f"  Filing lag         : {FILING_LAG_DAYS} days added to report_date")

    # Merge fundamentals (most recent report AVAILABLE before each monthly date)
    merged = pd.merge_asof(
        monthly.sort_values("date"),
        fund.sort_values("available_date"),
        by="ticker", left_on="date", right_on="available_date", direction="backward",
    )

    merged["market_cap"]     = merged["Close"] * merged["shares_outstanding"]
    merged["log_market_cap"] = np.log(merged["market_cap"].clip(lower=1))

    # ── Valuation metrics ────────────────────────────────────────────────────
    # The model has had no concept of cheap vs expensive. These two are
    # computable from fields already fetched in Stage 1.
    #
    #   price_to_book : market_cap / book value. Meaningless when equity is
    #                   negative, so those stay NaN rather than being inverted.
    #   cf_yield      : operating cash flow / market cap. Near-universal
    #                   coverage and much harder to manipulate than earnings.
    if "total_equity" in merged.columns:
        _eq = merged["total_equity"].replace(0, np.nan)
        merged["price_to_book"] = np.where(
            (_eq > 0) & merged["market_cap"].notna(),
            merged["market_cap"] / _eq,
            np.nan,
        )
        _pb_cov = merged["price_to_book"].notna().mean()
        print(f"  price_to_book      : computed, {_pb_cov:.0%} coverage")

    if "operating_cashflow" in merged.columns:
        merged["cf_yield"] = (
            merged["operating_cashflow"] / merged["market_cap"].replace(0, np.nan)
        )
        _cf_cov = merged["cf_yield"].notna().mean()
        print(f"  cf_yield           : computed, {_cf_cov:.0%} coverage")

    # Merge sentiment (most recent quarterly reading before each monthly date)
    SENT_COLS = ["sentiment_avg", "sentiment_bullish", "sentiment_bearish",
                 "sentiment_vol", "sentiment_count"] + [f"topic_{k}" for k in range(15)]
    try:
        sent = db_read("SELECT * FROM sentiment_features")
        sent["date"] = pd.to_datetime(sent["date"])
        sent = sent.sort_values(["ticker", "date"])
        merged = pd.merge_asof(
            merged.sort_values("date"),
            sent[["ticker", "date"] + SENT_COLS].sort_values("date"),
            by="ticker", on="date", direction="backward",
        )
        merged["sentiment_avg"]     = merged["sentiment_avg"].fillna(0.0)
        merged["sentiment_bullish"] = merged["sentiment_bullish"].fillna(0.5)
        merged["sentiment_bearish"] = merged["sentiment_bearish"].fillna(0.5)
        merged["sentiment_vol"]     = merged["sentiment_vol"].fillna(0.0)
        merged["sentiment_count"]   = merged["sentiment_count"].fillna(0.0)
        for _k in range(15):
            merged[f"topic_{_k}"] = merged[f"topic_{_k}"].fillna(0.0)
        print(f"  Sentiment merged — coverage: {merged[SENT_COLS[0]].notna().mean():.1%}")
        print(f"  Topic features   — topic_0 to topic_14 merged")
    except Exception as e:
        print(f"  Warning: sentiment skipped ({e}). Run Stage 5 first.")
        for col in SENT_COLS:
            merged[col] = 0.0

    # Filters — keep rows with NaN fwd_6m_return (most recent months = future predictions)
    merged = merged[merged["market_cap"] > MICROCAP_FLOOR]

    KEEP_COLS = [
        "ticker", "date", "Close", "Volume",
        "return", "mom_3m", "mom_6m", "mom_12m", "mom_12_2", "mom_12_2_rank", "mom_6m_rank", "vol_6m",
        "roa", "roe", "operating_margin",
        "debt_to_equity", "debt_to_assets", "current_ratio",
        "revenue_yoy", "income_yoy", "asset_growth", "accrual_ratio",
        "market_cap", "log_market_cap", "shares_outstanding",
        "total_equity", "operating_cashflow",
        "price_to_book", "cf_yield",
        "sentiment_avg", "sentiment_bullish", "sentiment_bearish",
        "sentiment_vol", "sentiment_count",
        *[f"topic_{k}" for k in range(15)],
        "fwd_6m_return",
    ]
    merged = merged[[c for c in KEEP_COLS if c in merged.columns]]

    # Impute nulls
    for col in ["revenue_yoy", "income_yoy", "asset_growth"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    for col in ["roa", "roe", "operating_margin", "debt_to_equity",
                "debt_to_assets", "current_ratio", "accrual_ratio"]:
        if col in merged.columns:
            merged[col] = merged.groupby("date")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Winsorize per date
    FACTOR_COLS = [
        "roa", "roe", "operating_margin", "debt_to_equity", "debt_to_assets",
        "current_ratio", "revenue_yoy", "income_yoy", "asset_growth", "accrual_ratio",
        "return", "mom_3m", "mom_6m", "mom_12m", "vol_6m", "log_market_cap",
        "price_to_book", "cf_yield",
    ]
    for col in FACTOR_COLS:
        if col in merged.columns:
            merged[col] = merged.groupby("date")[col].transform(
                lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
            )

    merged = merged.replace([np.inf, -np.inf], np.nan)

    db_write(merged, "model_dataset_clean")
    print(f"  ✓ model_dataset_clean  →  {len(merged):,} rows  |  {merged['ticker'].nunique()} tickers")
    print(f"    Columns: {list(merged.columns)}")
    null_pct = merged.isnull().mean()
    if null_pct.max() > 0.0:
        print("\n  Remaining nulls:")
        print(null_pct[null_pct > 0].to_string())


# ════════════════════════════════════════════════════════════════════════════════
# SHARED: FEATURE ENGINEERING (used by both Stage 7 and Stage 8)
# ════════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add engineered & sector-relative features. Returns (df, feature_cols)."""
    feature_cols = list(BASE_FEATURES)

    df["mom_reversal"]     = df["mom_3m"]  - df["mom_12m"]
    df["quality_spread"]   = df["roe"]     - df["roa"]
    df["earnings_quality"] = -df["accrual_ratio"]
    df["leverage_growth"]  = df["debt_to_equity"] * df["asset_growth"].fillna(0)
    df["mom_composite"]    = (df["mom_3m"] + df["mom_6m"] + df["mom_12m"]) / 3

    df["_debt_rank"] = df.groupby("date")["debt_to_assets"].rank(pct=True, ascending=False)
    df["roic_proxy"] = df["operating_margin"] * df["_debt_rank"]
    df = df.drop(columns=["_debt_rank"])

    ENGINEERED = ["mom_reversal", "quality_spread", "earnings_quality",
                  "leverage_growth", "mom_composite", "roic_proxy"]

    for col in ENGINEERED:
        df[col] = df.groupby("date")[col].transform(
            lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
        )

    feature_cols += ENGINEERED

    REL_COLS = ["mom_3m", "mom_6m", "mom_12m", "vol_6m", "log_market_cap",
                "roa", "roe", "mom_composite", "earnings_quality", "roic_proxy"]

    for col in REL_COLS:
        if col in df.columns:
            new_col = f"{col}_rel"
            df[new_col] = df.groupby(["date", "sector"])[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            feature_cols.append(new_col)

    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


def load_and_prepare(cutoff_date: str = None) -> tuple[pd.DataFrame, list[str]]:
    """Load model_dataset_clean, attach sectors, engineer features, build target."""
    df        = db_read("SELECT * FROM model_dataset_clean ORDER BY date, ticker")
    sector_df = db_read("SELECT * FROM ticker_sectors")

    df["date"] = pd.to_datetime(df["date"])

    if "fwd_6m_return" not in df.columns and "fwd_3m_return" in df.columns:
        df = df.rename(columns={"fwd_3m_return": "fwd_6m_return"})

    assert "fwd_6m_return" in df.columns

    if "Sector" in sector_df.columns:
        sector_df = sector_df.rename(columns={"Sector": "sector"})
    df = df.merge(sector_df[["ticker", "sector"]], on="ticker", how="left")
    df = df.dropna(subset=["sector"])

    if cutoff_date:
        df = df[df["date"] <= cutoff_date]

    df, feature_cols = engineer_features(df)

    sector_median    = df.groupby(["date", "sector"])["fwd_6m_return"].transform("median")
    df["outperforms"]= (df["fwd_6m_return"] > sector_median).astype(int)

    df = df.dropna(subset=["outperforms", "fwd_6m_return"]).sort_values("date").reset_index(drop=True)
    return df, feature_cols


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 7 — TRAIN MODEL (full dataset, current predictions)
# ════════════════════════════════════════════════════════════════════════════════

def train_model():
    banner("Train ML model on full dataset + generate current predictions", stage=7)

    df, FEATURE_COLS = load_and_prepare()

    print(f"  Rows     : {len(df):,}")
    print(f"  Tickers  : {df['ticker'].nunique()}")
    print(f"  Dates    : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Features : {len(FEATURE_COLS)}")
    print(f"\n  Sector distribution:")
    print(df["sector"].value_counts().to_string())
    print(f"\n  Outperformance rate: {df['outperforms'].mean():.1%}")

    # Thin-date filter
    date_counts = df.groupby("date")["ticker"].count()
    valid_dates = date_counts[date_counts >= THIN_DATE_MIN].index
    df = df[df["date"].isin(valid_dates)]
    print(f"\n  After thin-date filter (≥{THIN_DATE_MIN} tickers/date): {len(df):,} rows")

    X = df[FEATURE_COLS].values
    y = df["outperforms"].values

    # Time-decay weights — restored original formula (no clip, rescaled)
    # Matches old script: oldest~0.094, newest~2.8 giving richer recent emphasis
    max_date   = df["date"].max()
    months_ago = ((max_date - df["date"]).dt.days / 30.44)
    decay_rate = np.log(2) / HALF_LIFE_M
    weights    = np.exp(-decay_rate * months_ago)
    weights    = weights / weights.min() * 0.1   # rescale: oldest=0.1, newest scales proportionally
    print(f"\n  Time-decay weights: oldest={weights.min():.3f}, newest={weights.max():.3f}")

    # Optuna
    print(f"\n── Optuna ({N_OPTUNA} trials) ───────────────────────────────────────")
    inner_tscv = TimeSeriesSplit(n_splits=3)
    trial_aucs = []

    def objective(trial):
        params = {
            "max_iter":          trial.suggest_int("max_iter",          200, 600),
            "max_depth":         trial.suggest_int("max_depth",         3, 6),
            "learning_rate":     trial.suggest_float("learning_rate",   0.02, 0.12, log=True),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  10, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 0.8),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes",    15, 60),
            "random_state":      42,
        }
        aucs = []
        for tr, val in inner_tscv.split(X):
            m = HistGradientBoostingClassifier(**params)
            m.fit(X[tr], y[tr], sample_weight=weights.values[tr])
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        score = float(np.mean(aucs))
        trial_aucs.append(score)
        print(f"  Trial {len(trial_aucs):>2}/{N_OPTUNA}  AUC={score:.4f}  best={max(trial_aucs):.4f}")
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best_params = {**study.best_params, "random_state": 42}
    print(f"\n  Best inner AUC : {study.best_value:.4f}")
    print(f"  Params         : {best_params}")

    # Outer CV
    print(f"\n── Time-Series CV ({N_CV_FOLDS} folds) ────────────────────────────────")
    tscv    = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    cv_aucs = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        m = HistGradientBoostingClassifier(**best_params)
        m.fit(X[tr], y[tr], sample_weight=weights.values[tr])
        auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        cv_aucs.append(auc)
        print(f"  Fold {fold+1}  AUC={auc:.4f}  n_test={len(te):,}")
    print(f"\n  Mean AUC: {np.mean(cv_aucs):.4f}  ±  {np.std(cv_aucs):.4f}")

    # Final model
    final_model = HistGradientBoostingClassifier(**best_params)
    final_model.fit(X, y, sample_weight=weights.values)
    print("\n  ✓ Final model trained on all data.")

    # Feature importance
    last_tr, last_te = list(tscv.split(X))[-1]
    perm = permutation_importance(final_model, X[last_te], y[last_te],
                                  n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )
    print("\n── Top 20 Features ────────────────────────────────────────")
    print(importance_df.head(20).to_string(index=False))

    # Predictions on latest snapshot
    # Basic Materials and Consumer Defensive have sub-0.5 AUC — momentum
    # signals are anti-predictive there; exclude from live output
    EXCLUDED_SECTORS = {"Basic Materials", "Consumer Defensive"}
    latest_date = df["date"].max()
    latest = df[(df["date"] == latest_date) & (~df["sector"].isin(EXCLUDED_SECTORS))].copy()
    latest["outperform_prob"] = final_model.predict_proba(latest[FEATURE_COLS].values)[:, 1]

    for col in ["roic_proxy", "earnings_quality", "mom_6m"]:
        latest[f"{col}_display"] = latest.groupby("sector")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    print(f"\n── Predictions for {latest_date.date()} (6-Month Horizon) ─────────────")
    for sector in sorted(latest["sector"].dropna().unique()):
        top5 = latest[latest["sector"] == sector].sort_values("outperform_prob", ascending=False).head(5)
        print(f"\n  {sector}")
        print(f"  {'Ticker':<8} {'Prob':>6}  {'Mom6M':>7}  {'ROIC(z)':>8}  {'EarnQ(z)':>9}")
        print(f"  {'-'*48}")
        for _, row in top5.iterrows():
            print(
                f"  {row['ticker']:<8} "
                f"{row['outperform_prob']:>6.1%}  "
                f"{row['mom_6m']:>+7.1%}  "
                f"{row['roic_proxy_display']:>+8.2f}  "
                f"{row['earnings_quality_display']:>+9.2f}"
            )

    # Save
    PRED_COLS = ["ticker", "sector", "date", "outperform_prob",
                 "mom_3m", "mom_6m", "roa", "roe", "revenue_yoy", "market_cap",
                 "earnings_quality", "mom_composite", "roic_proxy"]
    PRED_COLS = [c for c in PRED_COLS if c in latest.columns]
    db_write(latest[PRED_COLS], "ml_predictions")

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "features": FEATURE_COLS, "best_params": best_params}, f)

    with open("cv_results.json", "w") as f:
        json.dump({
            "fold_aucs": cv_aucs, "mean_auc": float(np.mean(cv_aucs)),
            "std_auc": float(np.std(cv_aucs)), "n_features": len(FEATURE_COLS),
            "n_rows": len(df), "latest_date": str(latest_date.date()),
            "best_params": best_params, "optuna_inner_auc": float(study.best_value),
        }, f, indent=2)

    print("\n  ✓ ml_predictions  saved to DB")
    print("  ✓ model.pkl       saved")
    print("  ✓ cv_results.json saved")

    # Diagnostics chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.barplot(x=df["sector"].value_counts().index,
                y=df["sector"].value_counts().values, palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("Sector Distribution"); axes[0, 0].tick_params(axis="x", rotation=45)

    oc = df["outperforms"].value_counts()
    axes[0, 1].pie(oc.values, labels=["Does Not Outperform", "Outperforms"],
                   autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"], startangle=140)
    axes[0, 1].set_title("Outperformance Balance")

    folds = np.arange(1, len(cv_aucs) + 1)
    axes[1, 0].plot(folds, cv_aucs, marker="o", color="steelblue", label="Fold AUC")
    axes[1, 0].hlines(np.mean(cv_aucs), 1, len(cv_aucs), colors="r",
                      linestyles="dashed", label=f"Mean={np.mean(cv_aucs):.4f}")
    axes[1, 0].fill_between(folds, np.mean(cv_aucs)-np.std(cv_aucs),
                             np.mean(cv_aucs)+np.std(cv_aucs), color="r", alpha=0.15)
    axes[1, 0].set_title("CV AUC (8 Folds)"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.4)

    sns.barplot(x="importance", y="feature", data=importance_df.head(15), palette="magma", ax=axes[1, 1])
    axes[1, 1].set_title("Top 15 Feature Importances (Permutation)")

    plt.tight_layout()
    plt.savefig("model_diagnostics.png", dpi=150, bbox_inches="tight")
    print("  ✓ model_diagnostics.png saved")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 8 — BACKTEST EVALUATE (true OOS: train ≤ 2022, test 2023–2025)
# ════════════════════════════════════════════════════════════════════════════════

def backtest_evaluate():
    banner("True out-of-sample backtest  [Train ≤ 2022 | Test 2023–2025]", stage=8)

    # ── Load & prepare full dataset, but train on ≤ TRAIN_END ────────────────
    df_full, FEATURE_COLS = load_and_prepare()   # all dates, all engineering

    train_df = df_full[df_full["date"] <= TRAIN_END].copy()
    test_df  = df_full[(df_full["date"] >= TEST_START)].dropna(subset=["fwd_6m_return"]).copy()

    print(f"  TRAIN : {len(train_df):,} rows  |  {train_df['date'].min().date()} → {train_df['date'].max().date()}")
    print(f"  TEST  : {len(test_df):,}  rows  |  {test_df['date'].min().date()} → {test_df['date'].max().date()}")
    print(f"  Features: {len(FEATURE_COLS)}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["outperforms"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["outperforms"].values

    # Time-decay weights — restored original formula (no clip, rescaled)
    max_date   = train_df["date"].max()
    months_ago = ((max_date - train_df["date"]).dt.days / 30.44)
    decay_rate = np.log(2) / HALF_LIFE_M
    weights    = np.exp(-decay_rate * months_ago)
    weights    = weights / weights.min() * 0.1   # rescale: oldest=0.1, newest scales proportionally

    # Optuna (train data only)
    print(f"\n── Optuna ({N_OPTUNA} trials, train-only) ──────────────────────────")
    inner_tscv = TimeSeriesSplit(n_splits=3)
    trial_aucs = []

    def objective(trial):
        params = {
            "max_iter":          trial.suggest_int("max_iter",          200, 600),
            "max_depth":         trial.suggest_int("max_depth",         3, 6),
            "learning_rate":     trial.suggest_float("learning_rate",   0.02, 0.12, log=True),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  10, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 0.8),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes",    15, 60),
            "random_state":      42,
        }
        aucs = []
        for tr, val in inner_tscv.split(X_train):
            m = HistGradientBoostingClassifier(**params)
            m.fit(X_train[tr], y_train[tr], sample_weight=weights.values[tr])
            aucs.append(roc_auc_score(y_train[val], m.predict_proba(X_train[val])[:, 1]))
        score = float(np.mean(aucs))
        trial_aucs.append(score)
        print(f"  Trial {len(trial_aucs):>2}/{N_OPTUNA}  AUC={score:.4f}  best={max(trial_aucs):.4f}")
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best_params = {**study.best_params, "random_state": 42}
    print(f"\n  Best inner AUC : {study.best_value:.4f}")

    # Train final model on 2010–2022 only
    model = HistGradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train, sample_weight=weights.values)
    print("  ✓ Model trained exclusively on 2010–2022 data.")

    # Predictions
    test_df = test_df.copy()
    test_df["pred_prob"]    = model.predict_proba(X_test)[:, 1]
    test_df["pred_class"]   = (test_df["pred_prob"] >= 0.5).astype(int)
    test_df["actual_class"] = y_test
    test_df["correct"]      = test_df["pred_class"] == test_df["actual_class"]

    # ── Overall metrics ───────────────────────────────────────────────────────
    auc_oos   = roc_auc_score(y_test, test_df["pred_prob"])
    hit_rate  = test_df["correct"].mean()
    top_q     = test_df[test_df["pred_prob"] >= test_df["pred_prob"].quantile(0.80)]
    bot_q     = test_df[test_df["pred_prob"] <= test_df["pred_prob"].quantile(0.20)]
    top_ret   = top_q["fwd_6m_return"].mean()
    bot_ret   = bot_q["fwd_6m_return"].mean()
    all_ret   = test_df["fwd_6m_return"].mean()

    print(f"\n{'='*60}")
    print(f"  OUT-OF-SAMPLE RESULTS (2023–2025)")
    print(f"{'='*60}")
    print(f"  OOS AUC            : {auc_oos:.4f}")
    print(f"  Binary hit rate    : {hit_rate:.1%}")
    print(f"  Top-quintile return: {top_ret:+.2%}  avg 6M actual")
    print(f"  Bottom-quintile    : {bot_ret:+.2%}  avg 6M actual")
    print(f"  Market average     : {all_ret:+.2%}  avg 6M actual")
    print(f"  Long-short spread  : {top_ret - bot_ret:+.2%}")

    # Year breakdown
    print(f"\n── AUC by Year ──────────────────────────────────────────────")
    year_aucs = {}
    for year in sorted(test_df["date"].dt.year.unique()):
        yr = test_df[test_df["date"].dt.year == year]
        if len(yr) < 50 or yr["actual_class"].nunique() < 2:
            continue
        yr_auc = roc_auc_score(yr["actual_class"], yr["pred_prob"])
        yr_top = yr[yr["pred_prob"] >= yr["pred_prob"].quantile(0.80)]
        year_aucs[year] = yr_auc
        print(f"  {year}  AUC={yr_auc:.4f}  top-Q return={yr_top['fwd_6m_return'].mean():+.2%}  n={len(yr):,}")

    # Sector breakdown
    print(f"\n── AUC by Sector ─────────────────────────────────────────────")
    sector_metrics = []
    for sector in sorted(test_df["sector"].dropna().unique()):
        sec = test_df[test_df["sector"] == sector]
        if len(sec) < 30 or sec["actual_class"].nunique() < 2:
            continue
        sec_auc = roc_auc_score(sec["actual_class"], sec["pred_prob"])
        sec_top = sec[sec["pred_prob"] >= sec["pred_prob"].quantile(0.80)]
        sector_metrics.append({"sector": sector, "auc": sec_auc,
                                "top_q_return": sec_top["fwd_6m_return"].mean(), "n": len(sec)})
        print(f"  {sector:<30} AUC={sec_auc:.4f}  top-Q={sec_top['fwd_6m_return'].mean():+.2%}  n={len(sec):,}")

    # ── Side-by-side prediction vs reality ───────────────────────────────────
    for fallback_date in sorted(test_df["date"].unique(), reverse=True):
        snap = test_df[test_df["date"] == fallback_date].dropna(subset=["fwd_6m_return"])
        if len(snap) >= 20:
            latest_test_date = fallback_date
            break

    snap = snap.copy()
    snap["actual_outperform"] = snap["actual_class"].map({1: "✓ YES", 0: "✗ NO"})
    snap["pred_label"] = snap["pred_prob"].apply(
        lambda p: f"HIGH ({p:.0%})" if p >= HIGH_CONF_THRESHOLD else (f"MED  ({p:.0%})" if p >= 0.45 else f"LOW  ({p:.0%})")
    )
    snap["result_icon"] = snap["correct"].map({True: "✓", False: "✗"})

    print(f"\n\n{'='*82}")
    print(f"  PREDICTION vs REALITY  |  Snapshot: {latest_test_date.date()}  |  6-Month Window")
    print(f"{'='*82}")
    print(f"  {'Ticker':<7} {'Sector':<24} {'Model Pred':>12}  {'Actual 6M Ret':>13}  {'Outperform?':>11}  {'OK?':>4}")
    print(f"  {'-'*76}")

    total_correct = total_calls = 0
    high_conf_correct = high_conf_total = 0

    for sector in sorted(snap["sector"].dropna().unique()):
        sec_rows = snap[snap["sector"] == sector].sort_values("pred_prob", ascending=False).head(5)
        print(f"\n  ── {sector}")
        for _, row in sec_rows.iterrows():
            print(
                f"  {row['ticker']:<7} "
                f"{row['sector']:<24} "
                f"{row['pred_label']:>12}  "
                f"{row['fwd_6m_return']:>+12.1%}   "
                f"{row['actual_outperform']:>11}   "
                f"{row['result_icon']:>4}"
            )
            total_correct += int(row["correct"])
            total_calls   += 1
            if row["pred_prob"] >= HIGH_CONF_THRESHOLD:
                high_conf_correct += int(row["correct"])
                high_conf_total   += 1

    print(f"\n  Snapshot accuracy (all)        : {total_correct/total_calls:.1%}  ({total_correct}/{total_calls})")
    if high_conf_total:
        print(f"  High-confidence accuracy (≥65%): {high_conf_correct/high_conf_total:.1%}  ({high_conf_correct}/{high_conf_total})")

    # ── Feature importance (on OOS test set) ─────────────────────────────────
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_cols = [c for c in ["ticker", "sector", "date", "pred_prob", "pred_class",
                              "actual_class", "fwd_6m_return", "correct"] if c in test_df.columns]
    db_write(test_df[save_cols], "backtest_results")

    with open("backtest_report.txt", "w") as f:
        f.write("\n".join([
            "OUT-OF-SAMPLE BACKTEST REPORT",
            f"Train : 2010-01-01 → {TRAIN_END}",
            f"Test  : {TEST_START} → {test_df['date'].max().date()}",
            "",
            f"OOS AUC            : {auc_oos:.4f}",
            f"Binary hit rate    : {hit_rate:.1%}",
            f"Top-quintile return: {top_ret:+.2%}",
            f"Long-short spread  : {top_ret - bot_ret:+.2%}",
            "",
            "Best hyperparams:",
            json.dumps(best_params, indent=2),
        ]))

    # ── Diagnostics chart ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    G, R, B, O = "#2ecc71", "#e74c3c", "#3498db", "#e67e22"

    # 1. Prob distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(test_df[test_df["actual_class"]==1]["pred_prob"], bins=30, alpha=0.65, color=G, label="Outperformed")
    ax.hist(test_df[test_df["actual_class"]==0]["pred_prob"], bins=30, alpha=0.65, color=R, label="Did NOT")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Predicted Prob Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Outperform Probability"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. Quintile returns
    ax = fig.add_subplot(gs[0, 1])
    test_df["quintile"] = pd.qcut(test_df["pred_prob"], 5,
                                   labels=["Q1\n(Bear)", "Q2", "Q3", "Q4", "Q5\n(Bull)"])
    q_ret = test_df.groupby("quintile", observed=True)["fwd_6m_return"].mean() * 100
    cols  = [R if v < 0 else G for v in q_ret.values]
    bars  = ax.bar(q_ret.index.astype(str), q_ret.values, color=cols, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Avg 6M Return by Predicted Quintile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg 6M Return (%)"); ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, q_ret.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.2,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 3. AUC by year
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(list(year_aucs.keys()), list(year_aucs.values()), color=B, alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Random")
    ax.axhline(auc_oos, color=O, linewidth=1.5, label=f"Overall={auc_oos:.4f}")
    ax.set_title("AUC by Year (OOS)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.45, 0.70); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for yr, auc in year_aucs.items():
        ax.text(yr, auc+0.003, f"{auc:.3f}", ha="center", va="bottom", fontsize=8)

    # 4. Feature importances
    ax = fig.add_subplot(gs[1, :2])
    top20 = importance_df.head(20)
    ax.barh(top20["feature"][::-1], top20["importance"][::-1],
            color=[G if v > 0 else R for v in top20["importance"][::-1]])
    ax.set_title("Top 20 Feature Importances (Permutation, OOS)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Importance"); ax.grid(alpha=0.3, axis="x")

    # 5. AUC by sector
    ax = fig.add_subplot(gs[1, 2])
    if sector_metrics:
        sec_df = pd.DataFrame(sector_metrics).sort_values("auc", ascending=True)
        ax.barh(sec_df["sector"], sec_df["auc"],
                color=[G if a > 0.5 else R for a in sec_df["auc"]], alpha=0.8)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
        ax.set_title("OOS AUC by Sector", fontsize=11, fontweight="bold")
        ax.set_xlim(0.40, 0.75); ax.grid(alpha=0.3, axis="x")

    # 6. Cumulative return curves
    ax = fig.add_subplot(gs[2, :])
    dates = sorted(test_df["date"].unique())
    def cumret(mask):
        monthly = test_df[mask].groupby("date")["fwd_6m_return"].mean()
        return (1 + monthly.reindex(dates).fillna(0)).cumprod()

    top_mask = test_df["pred_prob"] >= test_df.groupby("date")["pred_prob"].transform(lambda x: x.quantile(0.80))
    bot_mask = test_df["pred_prob"] <= test_df.groupby("date")["pred_prob"].transform(lambda x: x.quantile(0.20))
    cum_top  = cumret(top_mask)
    cum_bot  = cumret(bot_mask)
    cum_all  = (1 + test_df.groupby("date")["fwd_6m_return"].mean().reindex(dates).fillna(0)).cumprod()

    ax.plot(dates, cum_top.values, color=G, linewidth=2,   label="Top Quintile (bullish)")
    ax.plot(dates, cum_bot.values, color=R, linewidth=2,   label="Bottom Quintile (bearish)")
    ax.plot(dates, cum_all.values, color=B, linewidth=1.5, linestyle="--", label="Market Avg")
    ax.fill_between(dates, cum_top.values, cum_all.values,
                    where=(cum_top.values > cum_all.values), alpha=0.12, color=G)
    ax.fill_between(dates, cum_bot.values, cum_all.values,
                    where=(cum_bot.values < cum_all.values), alpha=0.12, color=R)
    ax.set_title("Cumulative Performance: Top vs Bottom Quintile vs Market (OOS 2023–2025)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return (×1)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(
        f"True OOS Backtest  |  Train: 2010→2022  |  Test: 2023→2025  |  OOS AUC: {auc_oos:.4f}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.savefig("backtest_diagnostics.png", dpi=150, bbox_inches="tight")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BACKTEST FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  OOS AUC         : {auc_oos:.4f}", end="  →  ")
    if   auc_oos >= 0.60: print("✓ Strong signal")
    elif auc_oos >= 0.55: print("~ Decent signal")
    elif auc_oos >= 0.52: print("~ Weak signal")
    else:                 print("✗ Near-random")
    print(f"  Hit rate        : {hit_rate:.1%}")
    print(f"  Top-Q 6M return : {top_ret:+.2%}")
    print(f"  Long-short      : {top_ret - bot_ret:+.2%}")
    print(f"\n  Outputs:")
    print(f"    DB table  → backtest_results")
    print(f"    Report    → backtest_report.txt")
    print(f"    Chart     → backtest_diagnostics.png")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════════

STAGES = {
    1: ("fetch_financials",      fetch_financials),
    2: ("fetch_prices",          fetch_prices),
    3: ("build_fundamental",     build_fundamental_features),
    4: ("build_price_features",  build_price_features),
    5: ("build_sentiment",       build_sentiment),
    6: ("build_model_dataset",   build_model_dataset),
    7: ("train_model",           train_model),
    8: ("backtest_evaluate",     backtest_evaluate),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Alpha Pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="from_stage", type=int, metavar="N",
                       help="Run from stage N to end  (e.g. --from 3)")
    group.add_argument("--only", dest="only_stage", type=int, metavar="N",
                       help="Run only stage N          (e.g. --only 8)")
    args = parser.parse_args()

    if args.only_stage:
        stages_to_run = [args.only_stage]
    elif args.from_stage:
        stages_to_run = list(range(args.from_stage, max(STAGES) + 1))
    else:
        stages_to_run = list(STAGES.keys())

    print(f"\n{'═'*72}")
    print(f"  STOCK ALPHA PIPELINE  —  stages: {stages_to_run}")
    print(f"{'═'*72}")

    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            print(f"  Unknown stage {stage_num}, skipping.")
            continue
        name, fn = STAGES[stage_num]
        fn()

    print(f"\n{'═'*72}")
    print(f"  ALL DONE")
    print(f"{'═'*72}\n")
