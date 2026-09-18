"""
18_blind_validation.py
════════════════════════════════════════════════════════════════════════════════
BLIND HISTORICAL VALIDATION — fixed dates, no look-ahead, real outcomes.

Replay the model as it would have run on specific past dates, then check what
actually happened over the following six months.

FIXES vs the earlier version
────────────────────────────
  1. DATE ARITHMETIC. The old guard used pd.DateOffset(months=6), which turns
     2025-09-30 into 2026-03-30. The data is month-END, so the nearest date at
     or before that is 2026-02-28 — earlier than requested, so the guard fired
     and every snapshot was skipped. Navigation is now by POSITION in the
     sorted monthly date list: six months back is six entries back.

  2. SKIP DIAGNOSTICS. When a snapshot cannot be evaluated the script now says
     exactly why instead of printing "insufficient data".

  3. FEATURE SETS. BASE_FEATURES was 4 momentum columns, but engineer_features
     then derived quality_spread, earnings_quality, leverage_growth and
     roic_proxy from fundamentals, and added roa_rel / roe_rel / roic_proxy_rel
     on top — roughly 20 features, most of them fundamental. Three sets are now
     switchable so the comparison is honest.

NO LOOK-AHEAD: training stops six months before each snapshot, because a row
dated T only has its outcome known at T+6.

Usage:
  python3 18_blind_validation.py
  python3 18_blind_validation.py --features momentum_only
  python3 18_blind_validation.py --top 10
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/aditya/stock_alpha")
from db_config import db_read

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--features", default="all",
                choices=["momentum_only", "current_production", "full", "all"],
                help="which feature set to test; 'all' runs each in turn")
args = ap.parse_args()

TOP_N     = args.top
HORIZON_M = 6            # months held; also the training gap

SNAPSHOT_DATES = [
    "2021-09-30",
    "2022-09-30",
    "2023-09-30",
    "2024-09-30",
    "2025-09-30",
]

MODEL_PARAMS = dict(
    max_iter=300, max_depth=4, learning_rate=0.05,
    min_samples_leaf=30, l2_regularization=0.3,
    max_leaf_nodes=30, random_state=42,
)

# ── Three honest feature sets ────────────────────────────────────────────────
#
# momentum_only      price signals and nothing else. No engineered columns,
#                    no sector-relative fundamentals. A true 4-feature test.
#
# current_production what stock_alpha.py Stage 7 actually builds: 4 base
#                    momentum features PLUS engineered columns derived from
#                    fundamentals PLUS sector-relative versions. ~20 features,
#                    most of them fundamental despite the short BASE list.
#
# full               everything live_signals.py uses. ~38 features.
#
FEATURE_SETS = {
    "momentum_only": {
        "base":       ["mom_3m", "mom_6m", "mom_6m_rank", "vol_6m"],
        "engineered": [],
        "rel":        [],
    },
    "current_production": {
        "base":       ["mom_3m", "mom_6m", "mom_6m_rank", "vol_6m"],
        "engineered": ["mom_reversal", "quality_spread", "earnings_quality",
                       "leverage_growth", "mom_composite", "roic_proxy"],
        "rel":        ["mom_3m", "mom_6m", "mom_12m", "vol_6m", "log_market_cap",
                       "roa", "roe", "mom_composite", "earnings_quality",
                       "roic_proxy"],
    },
    "full": {
        "base":       ["mom_3m", "mom_6m", "mom_12m", "mom_12_2",
                       "mom_12_2_rank", "mom_6m_rank", "vol_6m", "return",
                       "roa", "roe", "operating_margin",
                       "debt_to_equity", "debt_to_assets", "current_ratio",
                       "revenue_yoy", "income_yoy", "asset_growth",
                       "accrual_ratio", "log_market_cap",
                       "price_to_book", "cf_yield"],
        "engineered": ["mom_reversal", "quality_spread", "earnings_quality",
                       "leverage_growth", "mom_composite", "roic_proxy"],
        "rel":        ["mom_3m", "mom_6m", "mom_12m", "mom_12_2", "vol_6m",
                       "log_market_cap", "roa", "roe", "mom_composite",
                       "earnings_quality", "roic_proxy"],
    },
}


def hr(t=""):
    print("\n" + "=" * 86)
    if t:
        print(f"  {t}")
        print("=" * 86)


# ════════════════════════════════════════════════════════════════════════════════
# LOAD + ENGINEER
# ════════════════════════════════════════════════════════════════════════════════

def load_data():
    df      = db_read("SELECT * FROM model_dataset_clean ORDER BY date, ticker")
    sectors = db_read("SELECT * FROM ticker_sectors")

    df["date"] = pd.to_datetime(df["date"])
    if "Sector" in sectors.columns:
        sectors = sectors.rename(columns={"Sector": "sector"})
    df = df.merge(sectors[["ticker", "sector"]], on="ticker", how="left")
    df = df.dropna(subset=["sector"])

    if "fwd_6m_return" not in df.columns:
        raise RuntimeError("fwd_6m_return missing from model_dataset_clean")

    # Engineered columns — built once, used by whichever set needs them.
    df["mom_reversal"]     = df["mom_3m"] - df["mom_12m"]
    df["quality_spread"]   = df["roe"]    - df["roa"]
    df["earnings_quality"] = -df["accrual_ratio"]
    df["leverage_growth"]  = df["debt_to_equity"] * df["asset_growth"].fillna(0)
    df["mom_composite"]    = (df["mom_3m"] + df["mom_6m"] + df["mom_12m"]) / 3
    df["_dr"] = df.groupby("date")["debt_to_assets"].rank(pct=True, ascending=False)
    df["roic_proxy"] = df["operating_margin"] * df["_dr"]
    df = df.drop(columns=["_dr"])

    for c in ["mom_reversal", "quality_spread", "earnings_quality",
              "leverage_growth", "mom_composite", "roic_proxy"]:
        df[c] = df.groupby("date")[c].transform(
            lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

    # Sector-relative versions of everything any set might ask for.
    for c in set(sum((s["rel"] for s in FEATURE_SETS.values()), [])):
        if c in df.columns:
            df[f"{c}_rel"] = df.groupby(["date", "sector"])[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8))

    # Label
    med = df.groupby(["date", "sector"])["fwd_6m_return"].transform("median")
    df["outperforms"] = (df["fwd_6m_return"] > med).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def build_feature_list(df, set_name):
    spec = FEATURE_SETS[set_name]
    cols = list(spec["base"]) + list(spec["engineered"]) \
         + [f"{c}_rel" for c in spec["rel"]]
    cols = list(dict.fromkeys(cols))
    return [c for c in cols if c in df.columns]


# ════════════════════════════════════════════════════════════════════════════════
# EVALUATE ONE SNAPSHOT
# ════════════════════════════════════════════════════════════════════════════════

def evaluate(df, features, requested, all_dates, date_pos):
    """
    Train on everything whose outcome was already known at `requested`,
    predict that snapshot, then score against the realised 6M return.
    """
    req = pd.Timestamp(requested)

    # Snapshot = latest month-end at or before the requested date, and it must
    # have realised outcomes (so it is at least HORIZON_M months in the past).
    usable = [d for d in all_dates
              if d <= req and df.loc[df["date"] == d, "fwd_6m_return"].notna().any()]
    if not usable:
        return None, f"no snapshot at/before {req.date()} has realised returns"
    snapshot = usable[-1]

    i = date_pos[snapshot]
    if i - HORIZON_M < 0:
        return None, (f"only {i} months of history before {snapshot.date()}, "
                      f"need {HORIZON_M}")

    # A row dated T has its outcome known at T+6, so training must stop six
    # entries before the snapshot. This is the no-look-ahead boundary.
    train_cutoff = all_dates[i - HORIZON_M]

    train = df[(df["date"] <= train_cutoff) & df["outperforms"].notna()].copy()
    snap  = df[(df["date"] == snapshot) & df["fwd_6m_return"].notna()].copy()

    if len(train) < 2000:
        return None, f"only {len(train):,} training rows"
    if len(snap) < 30:
        return None, f"only {len(snap)} stocks in the snapshot"

    # Thin-date filter, matching the production idea
    counts = train.groupby("date")["ticker"].count()
    train = train[train["date"].isin(counts[counts >= 30].index)]

    Xtr = train[features].replace([np.inf, -np.inf], np.nan)
    ytr = train["outperforms"].astype(int)
    Xsn = snap[features].replace([np.inf, -np.inf], np.nan)

    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(Xtr, ytr)

    snap = snap.copy()
    snap["ml_prob"] = model.predict_proba(Xsn)[:, 1]

    ml  = snap.nlargest(TOP_N, "ml_prob")
    mom = snap.dropna(subset=["mom_6m"]).nlargest(TOP_N, "mom_6m")

    uni_mean = snap["fwd_6m_return"].mean()
    uni_med  = snap["fwd_6m_return"].median()

    auc = np.nan
    if snap["outperforms"].nunique() == 2:
        auc = roc_auc_score(snap["outperforms"], snap["ml_prob"])

    return {
        "snapshot":     snapshot,
        "train_end":    train_cutoff,
        "train_rows":   len(train),
        "n_stocks":     len(snap),
        "ml_ret":       ml["fwd_6m_return"].mean(),
        "mom_ret":      mom["fwd_6m_return"].mean(),
        "uni_mean":     uni_mean,
        "uni_med":      uni_med,
        "ml_vs_uni":    ml["fwd_6m_return"].mean() - uni_mean,
        "mom_vs_uni":   mom["fwd_6m_return"].mean() - uni_mean,
        "ml_vs_mom":    ml["fwd_6m_return"].mean() - mom["fwd_6m_return"].mean(),
        "ml_worst":     ml["fwd_6m_return"].min(),
        "mom_worst":    mom["fwd_6m_return"].min(),
        "auc":          auc,
        "spearman":     snap[["ml_prob", "fwd_6m_return"]]
                            .corr(method="spearman").iloc[0, 1],
        "ml_tickers":   list(ml["ticker"]),
        "mom_tickers":  list(mom["ticker"]),
        "overlap":      len(set(ml["ticker"]) & set(mom["ticker"])),
    }, None


# ════════════════════════════════════════════════════════════════════════════════
# RUN ONE FEATURE SET
# ════════════════════════════════════════════════════════════════════════════════

def run_set(df, set_name, all_dates, date_pos, verbose=True):
    features = build_feature_list(df, set_name)

    hr(f"FEATURE SET: {set_name}   ({len(features)} features)")
    spec = FEATURE_SETS[set_name]
    print(f"\n  base       ({len(spec['base'])}): {', '.join(spec['base'])}")
    if spec["engineered"]:
        print(f"  engineered ({len(spec['engineered'])}): {', '.join(spec['engineered'])}")
    if spec["rel"]:
        print(f"  relative   ({len(spec['rel'])}): "
              f"{', '.join(c + '_rel' for c in spec['rel'])}")

    if set_name == "current_production":
        print("""
  Note: BASE_FEATURES is 4 momentum columns, but quality_spread,
  earnings_quality, leverage_growth and roic_proxy are all derived from
  fundamentals, and roa_rel / roe_rel / roic_proxy_rel add more. This set
  is mostly fundamental despite the short base list.""")

    rows = []
    for req in SNAPSHOT_DATES:
        res, why = evaluate(df, features, req, all_dates, date_pos)
        if res is None:
            print(f"\n  {req}  SKIPPED — {why}")
            continue
        rows.append(res)
        if verbose:
            print(f"""
  ── {req} ────────────────────────────────────────────────
     snapshot {res['snapshot'].date()}   train ends {res['train_end'].date()}   """
                  f"""{res['train_rows']:,} rows   {res['n_stocks']} stocks
     ML       {res['ml_ret']:>7.2%}   vs universe {res['ml_vs_uni']:>+7.2%}   """
                  f"""worst {res['ml_worst']:>7.2%}
     Momentum {res['mom_ret']:>7.2%}   vs universe {res['mom_vs_uni']:>+7.2%}   """
                  f"""worst {res['mom_worst']:>7.2%}
     Universe {res['uni_mean']:>7.2%}   AUC {res['auc']:.4f}   """
                  f"""rank corr {res['spearman']:>+.3f}   overlap {res['overlap']}/{TOP_N}
     ML  : {', '.join(res['ml_tickers'])}
     Mom : {', '.join(res['mom_tickers'])}""")

    if not rows:
        print("\n  No snapshot evaluated.")
        return None

    r = pd.DataFrame(rows)
    print(f"""
  ── Summary across {len(r)} snapshots ──────────────────────
     {'':<12} {'Mean':>8} {'vs universe':>12} {'Worst':>9} {'Beat universe':>15}
     {'-'*58}
     {'ML model':<12} {r['ml_ret'].mean():>7.2%} {r['ml_vs_uni'].mean():>+11.2%} """
          f"""{r['ml_worst'].mean():>8.2%} {(r['ml_vs_uni'] > 0).sum():>10}/{len(r)}
     {'Momentum':<12} {r['mom_ret'].mean():>7.2%} {r['mom_vs_uni'].mean():>+11.2%} """
          f"""{r['mom_worst'].mean():>8.2%} {(r['mom_vs_uni'] > 0).sum():>10}/{len(r)}
     {'Universe':<12} {r['uni_mean'].mean():>7.2%}

     Mean AUC {r['auc'].mean():.4f}   mean rank corr {r['spearman'].mean():+.3f}
     ML beat momentum in {(r['ml_vs_mom'] > 0).sum()}/{len(r)} snapshots""")

    r.insert(0, "feature_set", set_name)
    return r


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

hr("BLIND HISTORICAL VALIDATION")

print("\nLoading...")
df = load_data()

all_dates = sorted(df["date"].unique())
date_pos  = {d: i for i, d in enumerate(all_dates)}

realised = df.dropna(subset=["fwd_6m_return"])["date"]
print(f"  {len(df):,} rows  |  {df['ticker'].nunique()} tickers")
print(f"  Dates            : {all_dates[0].date()} -> {all_dates[-1].date()} "
      f"({len(all_dates)} month-ends)")
print(f"  Realised returns : up to {realised.max().date()}")
print(f"  Snapshots        : {', '.join(SNAPSHOT_DATES)}")
print(f"  Portfolio size   : top {TOP_N}, held {HORIZON_M} months")
print(f"""
  No look-ahead: for a snapshot at month i, training stops at month i-{HORIZON_M},
  because a row dated T only has its 6-month outcome known at T+{HORIZON_M}.""")

sets = list(FEATURE_SETS) if args.features == "all" else [args.features]
frames = [f for f in (run_set(df, s, all_dates, date_pos) for s in sets)
          if f is not None]

if len(frames) > 1:
    hr("FEATURE SETS COMPARED")
    comb = pd.concat(frames, ignore_index=True)
    g = comb.groupby("feature_set")
    print(f"\n  {'Feature set':<22} {'ML mean':>9} {'vs universe':>12} "
          f"{'AUC':>7} {'Worst':>9} {'Beat uni':>10}")
    print(f"  {'-'*74}")
    for name in sets:
        if name not in g.groups:
            continue
        s = g.get_group(name)
        print(f"  {name:<22} {s['ml_ret'].mean():>8.2%} "
              f"{s['ml_vs_uni'].mean():>+11.2%} {s['auc'].mean():>7.4f} "
              f"{s['ml_worst'].mean():>8.2%} "
              f"{(s['ml_vs_uni'] > 0).sum():>7}/{len(s)}")
    mom = frames[0]
    print(f"  {'(momentum baseline)':<22} {mom['mom_ret'].mean():>8.2%} "
          f"{mom['mom_vs_uni'].mean():>+11.2%} {'-':>7} "
          f"{mom['mom_worst'].mean():>8.2%}")

if frames:
    out = pd.concat(frames, ignore_index=True)
    out.to_csv("blind_validation.csv", index=False)
    print(f"\n  Saved -> blind_validation.csv  ({len(out)} rows)")

print(f"""
  Five snapshots is a small sample — treat any gap under a few percent as
  noise. These dates are now used; tuning on them and calling the same dates
  a blind test afterwards would not be honest.
""")
