"""
stock_alpha.py
════════════════════════════════════════════════════════════════════════════════
Data pipeline for the momentum screen. Collects only what live_signals.py and
the website actually use — nothing is kept for training or backtesting.

Stages:
  1  fetch_financials    → SEC EDGAR quarterly facts         → table: financials
  2  fetch_prices        → Yahoo Finance daily closes        → table: prices
  3  build_fundamentals  → one row per ticker, latest ratios → table: fundamentals

The ranking itself (6M-1M momentum, annualized volatility) is computed by
live_signals.py from `prices` + `fundamentals`.

Usage:
  python stock_alpha.py            # run all stages
  python stock_alpha.py --from 2   # resume from stage 2
  python stock_alpha.py --only 3   # run only stage 3
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from db_config import db_read, db_write

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

USER_AGENT      = "Adigateway adigateway@gmail.com"   # SEC requires a real contact
HEADERS         = {"User-Agent": USER_AGENT}
SEC_SLEEP       = 0.15          # seconds between SEC requests (≤10 req/s limit)

TODAY           = pd.Timestamp.today().normalize()
# Momentum needs ~7 month-ends and volatility needs 126 trading days, so two
# years of prices is plenty. Revenue YoY needs 5 quarters, so three years of
# filings covers it with room for late filers.
PRICE_START     = (TODAY - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
PRICE_END       = TODAY.strftime("%Y-%m-%d")
MIN_YEAR        = TODAY.year - 3

MICROCAP_FLOOR  = 300_000_000   # exclude market cap below this
FILING_LAG_DAYS = 45            # SEC "end" is the quarter END, not the filing
                                # date. A 10-Q is treated as public 45 days later.

# SEC XBRL tags → our column names. Only the fields behind a displayed ratio:
#   roa = net_income / total_assets        roe = net_income / total_equity
#   operating_margin = operating_income / revenue
#   revenue_yoy      = revenue growth over 4 quarters
#   cf_yield         = operating_cashflow / market_cap (needs shares_outstanding)
# Order matters: earlier tags win per-date, later tags fill gaps.
FIELD_MAP = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606, post-2018
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",                                             # legacy
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "operating_income":    ["OperatingIncomeLoss"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "total_assets":        ["Assets"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
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
# Income-statement concepts are kept to a 60-120 day window so a 10-K's
# full-year figure doesn't make Q4 look like a 4x jump.
# Cash-flow statements in 10-Qs are CUMULATIVE year-to-date, so they get a
# wider window and are annualised to a 365-day equivalent.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 120
PERIOD_RULES     = {"operating_cashflow": (60, 400)}
ANNUALIZE_FIELDS = {"operating_cashflow"}

RATIO_COLS = ["roa", "roe", "operating_margin", "revenue_yoy", "cf_yield"]

# ════════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def banner(title: str, stage: int = None):
    tag = f"[Stage {stage}]  " if stage else ""
    line = "═" * 72
    print(f"\n{line}")
    print(f"  {tag}{title}")
    print(f"{line}\n")

# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FETCH FINANCIALS (SEC EDGAR)
# ════════════════════════════════════════════════════════════════════════════════

def fetch_financials():
    banner("Fetch quarterly financials from SEC EDGAR", stage=1)

    # ── Ticker universe ──────────────────────────────────────────────────────
    # ticker_sectors is the stable universe. Wikipedia is a fallback only — its
    # page structure changes and would silently alter the universe.
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
        _tbl = next((t for t in pd.read_html(resp.text) if "Symbol" in t.columns), None)
        if _tbl is None:
            raise RuntimeError("No 'Symbol' table on the Wikipedia page. "
                               "Populate ticker_sectors instead.")
        tickers = _tbl["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        print(f"  S&P 500 tickers from Wikipedia : {len(tickers)}")

    cik_data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS).json()
    ticker_to_cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in cik_data.values()}
    ticker_cik_map = {t: ticker_to_cik[t] for t in tickers if t in ticker_to_cik}
    print(f"  CIK-matched tickers   : {len(ticker_cik_map)}")
    print(f"  Filings since         : {MIN_YEAR}")

    def _period_days(item: dict):
        """Days covered by a fact; None for instant concepts, -1 if unparseable."""
        start, end = item.get("start"), item.get("end")
        if not start or not end:
            return None
        try:
            return (pd.Timestamp(end) - pd.Timestamp(start)).days
        except Exception:
            return -1

    def extract_quarterly(facts: dict, ticker: str) -> list:
        """
        Pull quarterly fundamentals from a SEC companyfacts blob. Reads both
        us-gaap and dei, tries every tag per field (earlier wins per-date),
        and accepts 10-K filings so fiscal Q4 is not dropped.
        """
        all_facts = facts.get("facts", {})
        us_gaap   = all_facts.get("us-gaap", {})
        dei       = all_facts.get("dei", {})
        records: dict = {}

        for field, tags in FIELD_MAP.items():
            lo, hi = PERIOD_RULES.get(field, (QUARTER_MIN_DAYS, QUARTER_MAX_DAYS))
            for tag in tags:
                node = us_gaap.get(tag) or dei.get(tag)
                if node is None:
                    continue
                units = node.get("units", {})
                if not units:
                    continue
                data = (units.get("USD") or units.get("shares")
                        or units.get("USD/shares") or next(iter(units.values()), []))

                for item in data:
                    if item.get("form") not in ("10-Q", "10-K"):
                        continue
                    days = _period_days(item)
                    if days is not None and (days < 0 or not (lo <= days <= hi)):
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
                    if field in ANNUALIZE_FIELDS and days and days > 0:
                        val = val * 365.0 / days

                    rec = records.setdefault(date, {"ticker": ticker, "report_date": date})
                    if rec.get(field) is None:
                        rec[field] = val
                        rec[f"_{field}_days"] = days
                    elif field in ANNUALIZE_FIELDS and days:
                        # Prefer the longest period — less seasonal noise once annualised
                        if days > (rec.get(f"_{field}_days") or 0):
                            rec[field] = val
                            rec[f"_{field}_days"] = days
                # NO break — later tags fill remaining dates

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

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["ticker", "report_date"]).dropna(subset=["revenue"])

    per_ticker = df.groupby("ticker")["report_date"].max()
    stale = per_ticker[per_ticker < (TODAY - pd.Timedelta(days=200))]
    print(f"\n  ── Coverage ──────────────────────────────────────────────")
    for _fld in ("revenue", "net_income", "operating_cashflow", "shares_outstanding"):
        if _fld in df.columns:
            print(f"    {_fld:<20} present on {df[_fld].notna().mean():.0%} of rows")
    print(f"    Tickers stale >200 days    : {len(stale)} of {len(per_ticker)}")

    db_write(df, "financials", if_exists="replace")
    print(f"\n  ✓ financials  →  {len(df):,} rows  |  {df['ticker'].nunique()} tickers")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FETCH PRICES (Yahoo Finance)
# ════════════════════════════════════════════════════════════════════════════════

def fetch_prices():
    banner("Fetch daily closes from Yahoo Finance", stage=2)

    tickers = db_read("SELECT DISTINCT ticker FROM financials ORDER BY ticker")["ticker"].tolist()
    print(f"  Downloading {len(tickers)} tickers  |  {PRICE_START} → {PRICE_END}\n")

    raw = yf.download(
        tickers, start=PRICE_START, end=PRICE_END,
        progress=True, group_by="ticker", threads=True, auto_adjust=True,
    )

    frames = []
    for ticker in tickers:
        try:
            df = raw[ticker][["Close"]].copy() if len(tickers) > 1 else raw[["Close"]].copy()
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
    )[["ticker", "date", "Close"]]

    db_write(df_all, "prices", if_exists="replace")
    print(f"\n  ✓ prices  →  {len(df_all):,} rows  |  {df_all['ticker'].nunique()} tickers")
    print(f"    Date range: {df_all['date'].min().date()} → {df_all['date'].max().date()}")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BUILD FUNDAMENTALS SNAPSHOT (one row per ticker)
# ════════════════════════════════════════════════════════════════════════════════

def build_fundamentals():
    banner("Build latest fundamentals snapshot", stage=3)

    fin = db_read("SELECT * FROM financials")
    fin["report_date"] = pd.to_datetime(fin["report_date"])
    fin = fin.sort_values(["ticker", "report_date"])

    px = db_read('SELECT ticker, date, "Close" FROM prices')
    px["date"] = pd.to_datetime(px["date"], utc=True).dt.tz_localize(None)
    as_of = px["date"].max()
    last_close = px.sort_values("date").groupby("ticker")["Close"].last()

    # YoY growth needs the full quarterly history, so compute before filtering
    fin["revenue_yoy"] = fin.groupby("ticker")["revenue"].pct_change(4)

    # SEC omits shares on roughly half of filings; share counts move slowly,
    # so carry the last known value rather than dropping the company.
    fin["shares_outstanding"] = fin.groupby("ticker")["shares_outstanding"].ffill()

    # Latest quarter that was actually public on the snapshot date
    fin["available_date"] = fin["report_date"] + pd.Timedelta(days=FILING_LAG_DAYS)
    snap = fin[fin["available_date"] <= as_of].groupby("ticker").tail(1).copy()

    snap["Close"]      = snap["ticker"].map(last_close)
    snap["market_cap"] = snap["Close"] * snap["shares_outstanding"]
    snap = snap[snap["market_cap"] > MICROCAP_FLOOR]

    snap["roa"]              = snap["net_income"]       / snap["total_assets"]
    snap["roe"]              = snap["net_income"]       / snap["total_equity"]
    snap["operating_margin"] = snap["operating_income"] / snap["revenue"]
    snap["cf_yield"]         = snap["operating_cashflow"] / snap["market_cap"]
    snap = snap.replace([np.inf, -np.inf], np.nan)

    # Cross-sectional clip, then fill gaps (growth → 0, ratios → median)
    for col in RATIO_COLS:
        snap[col] = snap[col].clip(snap[col].quantile(0.01), snap[col].quantile(0.99))
    snap["revenue_yoy"] = snap["revenue_yoy"].fillna(0.0)
    for col in ["roa", "roe", "operating_margin", "cf_yield"]:
        snap[col] = snap[col].fillna(snap[col].median())

    snap["as_of"] = as_of
    out = snap[["ticker", "as_of", "report_date", "market_cap", *RATIO_COLS]]

    db_write(out, "fundamentals", if_exists="replace")
    print(f"  ✓ fundamentals  →  {len(out):,} tickers  |  as of {as_of.date()}")


# ════════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════════

STAGES = {
    1: ("fetch_financials",   fetch_financials),
    2: ("fetch_prices",       fetch_prices),
    3: ("build_fundamentals", build_fundamentals),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Alpha data pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="from_stage", type=int, metavar="N",
                       help="Run from stage N to end  (e.g. --from 2)")
    group.add_argument("--only", dest="only_stage", type=int, metavar="N",
                       help="Run only stage N          (e.g. --only 3)")
    args = parser.parse_args()

    if args.only_stage:
        stages_to_run = [args.only_stage]
    elif args.from_stage:
        stages_to_run = list(range(args.from_stage, max(STAGES) + 1))
    else:
        stages_to_run = list(STAGES.keys())

    print(f"\n{'═'*72}\n  STOCK ALPHA PIPELINE  —  stages: {stages_to_run}\n{'═'*72}")
    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            print(f"  Unknown stage {stage_num}, skipping.")
            continue
        STAGES[stage_num][1]()
    print(f"\n{'═'*72}\n  ALL DONE\n{'═'*72}\n")
