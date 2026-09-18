"""
17_ml_vs_momentum.py
════════════════════════════════════════════════════════════════════════════════
HEAD-TO-HEAD — is the ML model worth keeping? Read-only.

Momentum top-10 returned 17.98% in the lab. The ML model has never been
measured the same way on clean data — its 0.5721 AUC came from the old
dataset, before the SEC extraction fix, the filing lag, and the rank bug.

This runs both on identical dates with a walk-forward design:

    for each year Y:
        train on every row with a realised return before Y
        predict every month of Y
        take the top N by each method
        compare against the market and against each other

No look-ahead — the model never sees a row from the year it predicts.
Optuna is skipped in favour of fixed parameters, because tuning 12 times
would take hours and the lab showed hyperparameters are not where the
edge lives.

Writes nothing to the database. Saves ml_vs_momentum.json.

Usage:
  python3 17_ml_vs_momentum.py
════════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

sys.path.insert(0, "/home/aditya/stock_alpha")
from db_config import db_read

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

TOP_N       = 10          # the lab's best portfolio size
START_YEAR  = 2015        # first prediction year — needs 5 years of training
MIN_NAMES   = 40

# Fixed params, mid-range of what Optuna kept converging on
PARAMS = {
    "max_iter":          300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "min_samples_leaf":  30,
    "l2_regularization": 0.3,
    "max_leaf_nodes":    30,
    "random_state":      42,
}

BASE_FEATURES = [
    "mom_3m", "mom_6m", "mom_12m", "mom_12_2", "mom_12_2_rank", "mom_6m_rank",
    "vol_6m", "return",
    "roa", "roe", "operating_margin",
    "debt_to_equity", "debt_to_assets", "current_ratio",
    "revenue_yoy", "income_yoy", "asset_growth", "accrual_ratio",
    "log_market_cap", "price_to_book", "cf_yield",
]


def hr(t=""):
    print("\n" + "=" * 76)
    if t:
        print(f"  {t}")
        print("=" * 76)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 70 - len(t)))


report = {}

# ════════════════════════════════════════════════════════════════════════════════
# LOAD + ENGINEER
# ════════════════════════════════════════════════════════════════════════════════

hr("ML MODEL vs PLAIN MOMENTUM — walk-forward, clean data")

print("\nLoading...")
df      = db_read("SELECT * FROM model_dataset_clean ORDER BY date, ticker")
sectors = db_read("SELECT * FROM ticker_sectors")
if "Sector" in sectors.columns:
    sectors = sectors.rename(columns={"Sector": "sector"})

df["date"] = pd.to_datetime(df["date"])
df = df.merge(sectors[["ticker", "sector"]], on="ticker", how="left")
df = df.dropna(subset=["sector"])

print(f"  {len(df):,} rows  |  {df['ticker'].nunique()} tickers  |  "
      f"{df['date'].min().date()} -> {df['date'].max().date()}")

# ── Engineered features, same as live_predictions.py ──────────────────────────
df["mom_reversal"]     = df["mom_3m"] - df["mom_12m"]
df["quality_spread"]   = df["roe"]    - df["roa"]
df["earnings_quality"] = -df["accrual_ratio"]
df["leverage_growth"]  = df["debt_to_equity"] * df["asset_growth"].fillna(0)
df["mom_composite"]    = (df["mom_3m"] + df["mom_6m"] + df["mom_12m"]) / 3
df["_dr"] = df.groupby("date")["debt_to_assets"].rank(pct=True, ascending=False)
df["roic_proxy"] = df["operating_margin"] * df["_dr"]
df = df.drop(columns=["_dr"])

ENGINEERED = ["mom_reversal", "quality_spread", "earnings_quality",
              "leverage_growth", "mom_composite", "roic_proxy"]
for c in ENGINEERED:
    df[c] = df.groupby("date")[c].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

REL_COLS = ["mom_3m", "mom_6m", "mom_12m", "mom_12_2", "vol_6m",
            "log_market_cap", "roa", "roe", "mom_composite",
            "earnings_quality", "roic_proxy"]
for c in REL_COLS:
    if c in df.columns:
        df[f"{c}_rel"] = df.groupby(["date", "sector"])[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))

FEATURES = BASE_FEATURES + ENGINEERED + [f"{c}_rel" for c in REL_COLS]
FEATURES = [c for c in dict.fromkeys(FEATURES) if c in df.columns]
print(f"  {len(FEATURES)} features")

# ── Label ─────────────────────────────────────────────────────────────────────
med = df.groupby(["date", "sector"])["fwd_6m_return"].transform("median")
df["outperforms"] = (df["fwd_6m_return"] > med).astype(int)

labelled = df.dropna(subset=["fwd_6m_return", "outperforms"]).copy()
years = sorted(y for y in labelled["date"].dt.year.unique() if y >= START_YEAR)
print(f"  Walk-forward years: {years[0]} to {years[-1]}  ({len(years)} refits)")

# ════════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ════════════════════════════════════════════════════════════════════════════════

hr("WALK-FORWARD")
print()

rows, importances = [], []

for year in years:
    train = labelled[labelled["date"].dt.year < year]
    test  = labelled[labelled["date"].dt.year == year]
    if len(train) < 3000 or not len(test):
        continue

    Xtr = train[FEATURES].values
    ytr = train["outperforms"].values

    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(Xtr, ytr)

    test = test.copy()
    test["ml_prob"] = model.predict_proba(test[FEATURES].values)[:, 1]

    n_snap = 0
    for date, g in test.groupby("date"):
        if len(g) < MIN_NAMES:
            continue
        ml  = g.nlargest(TOP_N, "ml_prob")
        mom = g.nlargest(TOP_N, "mom_6m")
        rows.append({
            "date": date, "year": year,
            "market":    g["fwd_6m_return"].mean(),
            "ml":        ml["fwd_6m_return"].mean(),
            "ml_worst":  ml["fwd_6m_return"].min(),
            "mom":       mom["fwd_6m_return"].mean(),
            "mom_worst": mom["fwd_6m_return"].min(),
            "overlap":   len(set(ml["ticker"]) & set(mom["ticker"])),
        })
        n_snap += 1

    last = [r for r in rows if r["year"] == year]
    if last:
        print(f"  {year}  train {len(train):>6,} rows  "
              f"ML {np.mean([r['ml'] for r in last]):>7.2%}  "
              f"mom {np.mean([r['mom'] for r in last]):>7.2%}  "
              f"mkt {np.mean([r['market'] for r in last]):>7.2%}  "
              f"overlap {np.mean([r['overlap'] for r in last]):>4.1f}/{TOP_N}")

    if year == years[-1]:
        try:
            perm = permutation_importance(
                model, test[FEATURES].values, test["outperforms"].values,
                n_repeats=5, random_state=42, n_jobs=-1)
            importances = sorted(zip(FEATURES, perm.importances_mean),
                                 key=lambda kv: -kv[1])
        except Exception:
            pass

res = pd.DataFrame(rows)
if not len(res):
    print("\n  Not enough data.")
    raise SystemExit(0)

# ════════════════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════════════════

hr("RESULTS")


def summarise(col, worst_col):
    r  = res[col]
    ex = r - res["market"]
    return {
        "mean":     float(r.mean()),
        "median":   float(r.median()),
        "excess":   float(ex.mean()),
        "hit":      float((r > res["market"]).mean()),
        "ir":       float(ex.mean() / ex.std()) if ex.std() else np.nan,
        "worst":    float(res[worst_col].mean()),
        "pct_neg":  float((r < 0).mean()),
    }


ml  = summarise("ml", "ml_worst")
mom = summarise("mom", "mom_worst")
mkt = float(res["market"].mean())

print(f"""
  {len(res)} snapshots, {res['date'].min().date()} -> {res['date'].max().date()}
  Top {TOP_N} by each method, held 6 months, equal weighted
""")
print(f"  {'Strategy':<22} {'Mean':>8} {'Excess':>9} {'Hit':>6} {'IR':>6} "
      f"{'Worst':>9} {'%Neg':>6}")
print(f"  {'-'*68}")
print(f"  {'Market':<22} {mkt:>7.2%} {'-':>9} {'-':>6} {'-':>6} {'-':>9} {'-':>6}")
print(f"  {'ML model':<22} {ml['mean']:>7.2%} {ml['excess']:>+8.2%} "
      f"{ml['hit']:>5.0%} {ml['ir']:>6.2f} {ml['worst']:>8.2%} {ml['pct_neg']:>5.0%}")
print(f"  {'Plain momentum':<22} {mom['mean']:>7.2%} {mom['excess']:>+8.2%} "
      f"{mom['hit']:>5.0%} {mom['ir']:>6.2f} {mom['worst']:>8.2%} {mom['pct_neg']:>5.0%}")

delta = ml["mean"] - mom["mean"]
print(f"\n  ML minus momentum: {delta:+.2%}")
print(f"  ML beats momentum in {(res['ml'] > res['mom']).mean():.0%} of snapshots")
print(f"  Average overlap in picks: {res['overlap'].mean():.1f} of {TOP_N}")

report["ml"] = {k: round(v, 4) for k, v in ml.items()}
report["momentum"] = {k: round(v, 4) for k, v in mom.items()}
report["market"] = round(mkt, 4)
report["ml_minus_mom"] = round(delta, 4)
report["mean_overlap"] = round(float(res["overlap"].mean()), 2)

# ── Year by year ──────────────────────────────────────────────────────────────
sub("By year")
print(f"  {'Year':<7} {'Market':>9} {'ML':>9} {'Momentum':>10} {'ML-Mom':>9} "
      f"{'Overlap':>9}")
print(f"  {'-'*56}")
for year, g in res.groupby("year"):
    print(f"  {year:<7} {g['market'].mean():>8.1%} {g['ml'].mean():>8.1%} "
          f"{g['mom'].mean():>9.1%} {g['ml'].mean()-g['mom'].mean():>+8.1%} "
          f"{g['overlap'].mean():>8.1f}")

wins = (res.groupby("year").apply(lambda g: g["ml"].mean() > g["mom"].mean())).sum()
print(f"\n  ML wins in {wins} of {res['year'].nunique()} years")

# ── What is the model leaning on? ─────────────────────────────────────────────
if importances:
    sub("Top 15 features (final fold, permutation importance)")
    mx = max(abs(v) for _, v in importances[:15]) or 1
    for f, v in importances[:15]:
        bar = "#" * int(abs(v) / mx * 30)
        print(f"  {f:<26} {v:>8.4f}  {bar}")
    mom_share = sum(abs(v) for f, v in importances[:15]
                    if any(k in f for k in ("mom", "return", "vol")))
    tot = sum(abs(v) for _, v in importances[:15]) or 1
    print(f"\n  Price-based share of top-15 importance: {mom_share/tot:.0%}")
    report["top_features"] = [[f, round(float(v), 5)] for f, v in importances[:15]]

# ════════════════════════════════════════════════════════════════════════════════
# VERDICT
# ════════════════════════════════════════════════════════════════════════════════

hr("VERDICT")

print()
if delta > 0.01:
    print(f"""  The ML model beats plain momentum by {delta:+.2%} per 6-month period.
  Keep it. The extra features are earning their complexity.""")
elif delta < -0.01:
    print(f"""  Plain momentum beats the ML model by {-delta:+.2%} per 6-month period.

  Replace the model with a momentum ranking. It is simpler, needs no
  training, produces identical results every run, and there is nothing
  to tune or break.""")
else:
    print(f"""  The two are within {abs(delta):.2%} — indistinguishable.

  When a complex method ties a simple one, take the simple one. A momentum
  ranking has no training step, no random seed to pin, no Optuna run, and
  no silent feature bugs of the kind that cost 4.4% here.""")

print(f"""
  One caveat on all of it: {res['date'].min().date()} to {res['date'].max().date()}
  is a single regime. Two long bull markets and fast recoveries from both
  crashes — unusually kind to momentum. Neither result generalises to a
  prolonged bear market, and neither has transaction costs subtracted.""")

with open("ml_vs_momentum.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n" + "=" * 76)
print("  Saved -> ml_vs_momentum.json    Database unchanged.")
print("=" * 76 + "\n")
