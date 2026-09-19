"""
live_signals.py
════════════════════════════════════════════════════════════════════════════════
PRODUCTION — Risk-Adjusted Momentum with Hard Profitability Gate.

FEATURES:
1. Hard Gate: Drops all companies with ROA <= 0 BEFORE ranking.
2. Signal: Risk-adjusted 6M-1M price return (6-month return skipping latest month,
   scaled by 6-month annualized daily price volatility).

Usage:
  python3 live_signals.py
  python3 live_signals.py --top 10
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/aditya/stock_alpha")
from db_config import db_read, db_write, get_engine

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--allow-unprofitable", action="store_true",
                help="bypass the hard gate and include ROA <= 0 companies")
args = ap.parse_args()

TOP_N    = args.top
LOOKBACK = 6      # months


def hr(t=""):
    print("\n" + "=" * 78)
    if t:
        print(f"  {t}")
        print("=" * 78)


def cap(v):
    if pd.isna(v):
        return "n/a"
    return f"${v/1e12:.2f}T" if v >= 1e12 else f"${v/1e9:.1f}B"


# ════════════════════════════════════════════════════════════════════════════════
# LOAD & CALCULATE
# ════════════════════════════════════════════════════════════════════════════════

hr("LIVE SIGNALS — Risk-Adjusted Momentum (6M-1M)")
print(f"\n  Run date : {date.today()}")
print(f"  Signal   : Risk-adjusted 6M-1M price return (Return / Volatility)")
print("\nLoading...")

# Fundamentals and sector — for screening the output, NOT for ranking
# (built by stock_alpha.py stage 3: one row per ticker)
snap_df = db_read("""
    SELECT ticker, as_of AS date, roa, roe, operating_margin, revenue_yoy,
           market_cap, cf_yield
    FROM fundamentals
""")
sectors = db_read("SELECT * FROM ticker_sectors")
if "Sector" in sectors.columns:
    sectors = sectors.rename(columns={"Sector": "sector"})

snapshot = pd.to_datetime(snap_df["date"].iloc[0]).date()
horizon  = (pd.Timestamp(snapshot) + pd.DateOffset(months=6)).date()

# Raw prices — true momentum
px = db_read('SELECT ticker, date, "Close" FROM prices')
px["date"] = pd.to_datetime(px["date"], utc=True).dt.tz_localize(None)
px = px[px["date"] <= pd.Timestamp(snapshot)]
px = px.sort_values(["ticker", "date"])

# 1. Resample to monthly closes
monthly = (px.set_index(["ticker", "date"])
             .groupby(level=0).resample("ME", level=1).last()
             .reset_index())

# 2. Calculate 6M-1M return (t-7 to t-1 to cover a full 6-month window)
monthly["price_1m_ago"] = monthly.groupby("ticker")["Close"].shift(1)
monthly["price_7m_ago"] = monthly.groupby("ticker")["Close"].shift(LOOKBACK + 1)
monthly["mom_6m_true"]  = (monthly["price_1m_ago"] / monthly["price_7m_ago"]) - 1.0

# 3. Calculate 6-month daily return volatility (~126 trading days).
#    Annualized below; it is also the σ the website's Monte Carlo uses.
px["daily_return"] = px.groupby("ticker")["Close"].pct_change()
px["daily_vol_6m"]  = px.groupby("ticker")["daily_return"].transform(lambda x: x.rolling(126).std())

latest_vol = px.groupby("ticker").tail(1)[["ticker", "daily_vol_6m"]]
latest_px  = monthly.groupby("ticker").tail(1)[["ticker", "Close", "mom_6m_true"]]

# 4. Merge dataset
df = (snap_df.merge(latest_px, on="ticker", how="left")
             .merge(latest_vol, on="ticker", how="left")
             .merge(sectors[["ticker", "sector"]], on="ticker", how="left")
             .dropna(subset=["sector", "mom_6m_true", "daily_vol_6m"]))

# 5. HARD GATE: Filter out money-losing companies (ROA <= 0) before ranking
if not args.allow_unprofitable:
    before_gate = len(df)
    df = df[df["roa"] > 0]
    dropped = before_gate - len(df)
    print(f"  Hard Gate : Dropped {dropped} unprofitable tickers (ROA <= 0)")

df["vol_annualized"] = df["daily_vol_6m"] * np.sqrt(252)

df["mom_score_risk_adj"] = np.where(
    df["vol_annualized"] > 0, 
    df["mom_6m_true"] / df["vol_annualized"], 
    0
)

print(f"  Snapshot  : {snapshot}   ({len(df)} tickers remaining)")
print(f"  Horizon   : {snapshot} -> {horizon}")

# ════════════════════════════════════════════════════════════════════════════════
# RANK
# ════════════════════════════════════════════════════════════════════════════════

df["mom_rank"] = df["mom_score_risk_adj"].rank(ascending=False).astype(int)
df["mom_pct"]  = df["mom_score_risk_adj"].rank(pct=True)
df["signal"]   = np.where(df["mom_rank"] <= TOP_N, "BUY",
                  np.where(df["mom_pct"] >= 0.5, "UPPER", "LOWER"))

top = df.nsmallest(TOP_N, "mom_rank")

hr(f"TOP {TOP_N} BY RISK-ADJUSTED 6M MOMENTUM (HARD GATED)")
print(f"\n  {'#':<3} {'Ticker':<7} {'Sector':<20} {'6M mom':>8} {'Ann Vol':>8} {'Risk-Adj':>9} "
      f"{'ROA':>7} {'Mkt cap':>9}  Profitable")
print(f"  {'-'*98}")
for i, (_, r) in enumerate(top.iterrows(), 1):
    roa = r.get("roa")
    roa = 0.0 if pd.isna(roa) else roa
    flag = "yes" if roa > 0.03 else "marginal"
    print(f"  {i:<3} {r['ticker']:<7} {str(r['sector'])[:19]:<20} "
          f"{r['mom_6m_true']*100:>7.1f}% "
          f"{r['vol_annualized']*100:>7.1f}% "
          f"{r['mom_score_risk_adj']:>9.2f} "
          f"{roa*100:>6.1f}% "
          f"{cap(r.get('market_cap')):>9}  {flag}")

solid = int((top["roa"] > 0.03).sum())
thin  = int(((top["roa"] > 0) & (top["roa"] <= 0.03)).sum())

print(f"""
  Profitability Breakdown ({TOP_N} picks):
    solidly profitable (ROA > 3%)  {solid}
    marginal (0-3%)                {thin}
    losing money (ROA <= 0%)       0  (Hard Gated)""")

sec = top["sector"].value_counts()
print(f"\n  Sector mix:")
for s, n in sec.items():
    print(f"    {s:<24} {n:>2}  {'#' * n}")

# ════════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════════

hr("SAVING")

# Everything the API/website reads. mom_6m_true (μ) and vol_annualized (σ)
# are the two inputs to the portfolio Monte Carlo simulation.
out = (
    df[[
        "ticker", "sector", "date", "mom_6m_true", "vol_annualized", "mom_score_risk_adj",
        "mom_rank", "mom_pct", "signal", "roa", "roe", "operating_margin",
        "revenue_yoy", "market_cap", "cf_yield", "Close"
    ]]
    .sort_values("mom_rank")
)

db_write(out, "live_signals", if_exists="replace")
print(f"\n  live_signals      -> {len(out):,} rows")

try:
    arch = out.copy()
    arch["run_date"] = str(date.today())
    arch[["run_date", "ticker", "sector", "mom_6m_true", "vol_annualized", 
          "mom_score_risk_adj", "mom_rank", "signal", "roa", "revenue_yoy", "market_cap"]].to_sql(
        "prediction_runs", get_engine(), if_exists="append", index=False)
    print(f"  prediction_runs   -> archived")
except Exception as e:
    print(f"  prediction_runs   -> skipped ({str(e)[:70]})")

with open("live_signals.json", "w") as f:
    json.dump({
        "run_date":         str(date.today()),
        "snapshot":         str(snapshot),
        "horizon":          str(horizon),
        "strategy":         "risk_adjusted_momentum_6m_1m",
        "feature":          "mom_score_risk_adj",
        "hard_gate":        "ROA > 0",
        "top_n":            TOP_N,
        "n_universe":       int(len(df)),
        "picks":            top["ticker"].tolist(),
    }, f, indent=2)
print(f"  live_signals.json -> saved")

hr("DONE")