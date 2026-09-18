"""
live_signals.py
════════════════════════════════════════════════════════════════════════════════
PRODUCTION — momentum only.

ONE FEATURE: mom_6m, the 6-month price return. Rank, take the top N, hold six
months. No model, no training, no hyperparameters, nothing to tune.

WHY THE ML MODEL IS GONE
────────────────────────
Blind validation across five snapshots (18_blind_validation.py) with a proper
six-month training gap:

    momentum_only        AUC 0.4999   rank corr +0.007
    current_production   AUC 0.5089   rank corr +0.019
    full (38 features)   AUC 0.5034   rank corr -0.002

AUC 0.50 over ~1,375 stock-outcome pairs is a coin flip. The three feature
sets also picked almost entirely different stocks on the same date, which is
what fitting noise looks like. The earlier 17.92% walk-forward result came
from a look-ahead bug: it trained on rows whose outcomes were not yet known.

Momentum is not trained, so it cannot leak. Its edge survived the clean test:
24.16% mean versus a 9.91% universe, beating it in 3 of 5 snapshots.

MOMENTUM IS COMPUTED FROM RAW PRICES HERE
─────────────────────────────────────────
model_dataset_clean winsorises mom_6m at the 99th percentile, which flattens
the top three or four names to an identical value. Acceptable for training,
wrong for ranking and misleading on screen. This reads the prices table.

Writes: live_signals, live_predictions (website), prediction_runs, JSON.

Usage:
  python3 live_signals.py
  python3 live_signals.py --top 10
  python3 live_signals.py --profitable-only
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
ap.add_argument("--profitable-only", action="store_true",
                help="drop companies with ROA <= 0 before ranking")
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
# LOAD
# ════════════════════════════════════════════════════════════════════════════════

hr("LIVE SIGNALS — momentum")
print(f"\n  Run date : {date.today()}")
print(f"  Signal   : 6-month price return. One feature. No model.")

print("\nLoading...")

# Fundamentals and sector — for screening the output, NOT for ranking
snap_df = db_read("""
    SELECT ticker, date, roa, roe, operating_margin, revenue_yoy,
           market_cap, price_to_book, cf_yield, vol_6m
    FROM model_dataset_clean
    WHERE date = (SELECT MAX(date) FROM model_dataset_clean)
""")
sectors = db_read("SELECT * FROM ticker_sectors")
if "Sector" in sectors.columns:
    sectors = sectors.rename(columns={"Sector": "sector"})

snapshot = pd.to_datetime(snap_df["date"].iloc[0]).date()
horizon  = (pd.Timestamp(snapshot) + pd.DateOffset(months=6)).date()

# Raw prices — true momentum, unwinsorised
px = db_read('SELECT ticker, date, "Close" FROM prices')
px["date"] = pd.to_datetime(px["date"], utc=True).dt.tz_localize(None)
px = px[px["date"] <= pd.Timestamp(snapshot)]

monthly = (px.sort_values(["ticker", "date"])
             .set_index(["ticker", "date"])
             .groupby(level=0).resample("ME", level=1).last()
             .reset_index())
monthly["mom_6m_true"] = monthly.groupby("ticker")["Close"].pct_change(LOOKBACK)

latest_px = (monthly.sort_values("date")
                    .groupby("ticker").tail(1)[["ticker", "Close", "mom_6m_true"]])

df = (snap_df.merge(latest_px, on="ticker", how="left")
             .merge(sectors[["ticker", "sector"]], on="ticker", how="left")
             .dropna(subset=["sector", "mom_6m_true"]))

print(f"  Snapshot : {snapshot}   ({len(df)} tickers)")
print(f"  Horizon  : {snapshot} -> {horizon}")

if args.profitable_only:
    before = len(df)
    df = df[df["roa"] > 0]
    print(f"  Filter   : ROA > 0 — {before} -> {len(df)} tickers")

# ════════════════════════════════════════════════════════════════════════════════
# RANK
# ════════════════════════════════════════════════════════════════════════════════

df["mom_rank"] = df["mom_6m_true"].rank(ascending=False).astype(int)
df["mom_pct"]  = df["mom_6m_true"].rank(pct=True)
df["signal"]   = np.where(df["mom_rank"] <= TOP_N, "BUY",
                  np.where(df["mom_pct"] >= 0.5, "UPPER", "LOWER"))

top = df.nsmallest(TOP_N, "mom_rank")

hr(f"TOP {TOP_N} BY 6-MONTH MOMENTUM")
print(f"\n  {'#':<3} {'Ticker':<7} {'Sector':<22} {'6M mom':>9} {'ROA':>7} "
      f"{'Rev YoY':>8} {'CF yld':>7} {'Mkt cap':>9}  Profitable")
print(f"  {'-'*94}")
for i, (_, r) in enumerate(top.iterrows(), 1):
    roa = r.get("roa")
    roa = 0.0 if pd.isna(roa) else roa
    flag = "yes" if roa > 0.03 else ("marginal" if roa > 0 else "NO - losing money")
    print(f"  {i:<3} {r['ticker']:<7} {str(r['sector'])[:21]:<22} "
          f"{r['mom_6m_true']*100:>8.1f}% "
          f"{roa*100:>6.1f}% "
          f"{(r.get('revenue_yoy') or 0)*100:>7.1f}% "
          f"{(r.get('cf_yield') or 0)*100:>6.1f}% "
          f"{cap(r.get('market_cap')):>9}  {flag}")

solid  = int((top["roa"] > 0.03).sum())
thin   = int(((top["roa"] > 0) & (top["roa"] <= 0.03)).sum())
losing = int((top["roa"] <= 0).sum())

print(f"""
  Profitability of these {TOP_N}:
    solidly profitable (ROA > 3%)  {solid}
    marginal (0-3%)                {thin}
    losing money                   {losing}

  Momentum ranks by price, not quality. Check the money-losers yourself.""")

sec = top["sector"].value_counts()
print(f"\n  Sector mix:")
for s, n in sec.items():
    print(f"    {s:<24} {n:>2}  {'#' * n}")
if len(sec) and sec.iloc[0] / TOP_N > 0.4:
    print(f"""
  {sec.index[0]} is {sec.iloc[0]/TOP_N:.0%} of the list. Momentum concentrates
  by nature — when one sector runs it fills the ranking. Size accordingly.""")

# ════════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════════

hr("SAVING")

out = df[["ticker", "sector", "date", "mom_6m_true", "mom_rank", "mom_pct",
          "signal", "roa", "roe", "operating_margin", "revenue_yoy",
          "market_cap", "price_to_book", "cf_yield", "vol_6m", "Close"]] \
        .sort_values("mom_rank")

db_write(out, "live_signals", if_exists="replace")
print(f"\n  live_signals      -> {len(out):,} rows")

# Website compatibility — it reads outperform_prob
compat = out.copy()
compat["outperform_prob"] = compat["mom_pct"]
compat["mom_6m"] = compat["mom_6m_true"]
db_write(compat[["ticker", "sector", "date", "outperform_prob", "mom_6m",
                 "roa", "roe", "operating_margin", "revenue_yoy",
                 "market_cap", "price_to_book", "cf_yield"]],
         "live_predictions", if_exists="replace")
print(f"  live_predictions  -> {len(compat):,} rows  (website)")

try:
    arch = out.copy()
    arch["run_date"] = str(date.today())
    arch[["run_date", "ticker", "sector", "mom_6m_true", "mom_rank",
          "signal", "roa", "revenue_yoy", "market_cap"]].to_sql(
        "prediction_runs", get_engine(), if_exists="append", index=False)
    print(f"  prediction_runs   -> archived")
except Exception as e:
    print(f"  prediction_runs   -> skipped ({str(e)[:70]})")

with open("live_signals.json", "w") as f:
    json.dump({
        "run_date":        str(date.today()),
        "snapshot":        str(snapshot),
        "horizon":         str(horizon),
        "strategy":        "momentum_6m",
        "feature":         "mom_6m",
        "top_n":           TOP_N,
        "n_universe":      int(len(df)),
        "picks":           top["ticker"].tolist(),
        "profitable_only": bool(args.profitable_only),
    }, f, indent=2)
print(f"  live_signals.json -> saved")

# ════════════════════════════════════════════════════════════════════════════════

hr("RECORD")

print("""
  Blind validation, 5 snapshots 2021-2025, top 15 held 6 months:

    Momentum   24.16%   vs universe +14.25%   beat it 3 of 5
    Universe    9.91%

  Per snapshot, versus the universe:

    2021   -11.66%
    2022    -2.26%
    2023   +26.07%
    2024    +8.60%
    2025   +50.50%

  Two of five were negative, and 2025 carries most of the average — that was
  the AI and semiconductor rally. Excluding it the edge is roughly +5%.

  So: real, but modest in ordinary years and heavily regime-dependent. Best in
  trending markets, worst in choppy ones. Transaction costs are in none of
  these numbers, and turnover runs near 40% a month.

  Rerun after each quarterly pipeline, or any time. Output is deterministic.
""")

print("=" * 78)
print("  Done.")
print("=" * 78 + "\n")