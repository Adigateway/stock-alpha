"""
16_momentum_lab.py
════════════════════════════════════════════════════════════════════════════════
MOMENTUM LAB — pure price signal. Read-only.

Three separate tests have now shown fundamental screening does not work on
this universe:

    gates only          8.74%  vs market 9.36%
    gates + quality    10.28%
    veto 2+ failures   12.60%  vs momentum 13.82%
    momentum alone     13.82%

So this script drops fundamentals entirely and asks the only remaining
question: which momentum is best?

  TEST 1  HORIZON        3m / 6m / 12m / 12-2 / composite
  TEST 2  RANKING        raw value / cross-sectional / sector-relative
  TEST 3  RISK ADJUST    momentum divided by volatility
  TEST 4  BLENDS         averaging two horizons
  TEST 5  SIZE           10 / 20 / 30 / 50 names
  TEST 6  TURNOVER       how much churn each variant needs

Note: mom_6m_rank and mom_12_2_rank in the database are ranked WITHIN ticker
(each stock against its own history), not across tickers on a date. This
script computes cross-sectional ranks itself and ignores those columns.

Writes nothing to the database. Saves momentum_lab.json.

Usage:
  python3 16_momentum_lab.py
════════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/aditya/stock_alpha")
from db_config import db_read

TOP_N     = 20      # default portfolio size
MIN_NAMES = 40      # skip thin snapshots
SIZES     = [10, 20, 30, 50]


def hr(t=""):
    print("\n" + "=" * 76)
    if t:
        print(f"  {t}")
        print("=" * 76)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 70 - len(t)))


report = {}

# ════════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════════

hr("MOMENTUM LAB — pure price signal")

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

# ════════════════════════════════════════════════════════════════════════════════
# BUILD SIGNALS
# ════════════════════════════════════════════════════════════════════════════════

# vol_6m was log-transformed in Stage 4; exponentiate to recover the actual
# standard deviation for risk adjustment.
df["vol_raw"] = np.exp(df["vol_6m"]).clip(lower=1e-6)

# Composite of the three plain horizons
df["mom_avg"] = df[["mom_3m", "mom_6m", "mom_12m"]].mean(axis=1)

# Risk-adjusted: return per unit of volatility
df["mom_6m_vadj"]   = df["mom_6m"]   / df["vol_raw"]
df["mom_12_2_vadj"] = df["mom_12_2"] / df["vol_raw"]

# Cross-sectional percentile ranks (each stock against every other stock
# on the same date) — this is what the database columns should have been.
for c in ["mom_3m", "mom_6m", "mom_12m", "mom_12_2", "mom_avg",
          "mom_6m_vadj", "mom_12_2_vadj"]:
    df[f"{c}_x"] = df.groupby("date")[c].rank(pct=True)

# Sector-relative z-scores
for c in ["mom_6m", "mom_12_2"]:
    df[f"{c}_sec"] = df.groupby(["date", "sector"])[c].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

# Blends of two horizons, as an average of their cross-sectional ranks
df["blend_6_122"] = (df["mom_6m_x"] + df["mom_12_2_x"]) / 2
df["blend_3_12"]  = (df["mom_3m_x"] + df["mom_12m_x"]) / 2
df["blend_vadj"]  = (df["mom_6m_x"] + df["mom_6m_vadj_x"]) / 2

bt = df.dropna(subset=["fwd_6m_return"]).copy()
print(f"  Backtest rows: {len(bt):,} across {bt['date'].nunique()} snapshots")

# ════════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def run(signal, top_n=TOP_N, ascending=False):
    """
    Rank by `signal` each month, take the top `top_n`, hold 6 months.
    Returns a per-snapshot DataFrame plus a turnover figure.
    """
    rows, prev = [], None
    turnovers  = []

    for date, g in bt.groupby("date"):
        if len(g) < MIN_NAMES or signal not in g.columns:
            continue
        gg = g.dropna(subset=[signal])
        if len(gg) < top_n:
            continue

        sel = gg.nsmallest(top_n, signal) if ascending else gg.nlargest(top_n, signal)
        held = set(sel["ticker"])

        if prev is not None:
            turnovers.append(len(held - prev) / top_n)
        prev = held

        r = sel["fwd_6m_return"]
        rows.append({
            "date":   date,
            "ret":    r.mean(),
            "worst":  r.min(),
            "best":   r.max(),
            "market": g["fwd_6m_return"].mean(),
        })

    if not rows:
        return None, np.nan
    return pd.DataFrame(rows), (np.mean(turnovers) if turnovers else np.nan)


def stats(res, turnover=np.nan):
    """Summarise one strategy."""
    if res is None or not len(res):
        return None
    excess = res["ret"] - res["market"]
    return {
        "mean":     float(res["ret"].mean()),
        "median":   float(res["ret"].median()),
        "excess":   float(excess.mean()),
        "ir":       float(excess.mean() / excess.std()) if excess.std() else np.nan,
        "hit":      float((res["ret"] > res["market"]).mean()),
        "worst_hold": float(res["worst"].mean()),
        "worst_snap": float(res["ret"].min()),
        "turnover": float(turnover),
        "n":        int(len(res)),
    }


def table(results, title, note=""):
    """Print a comparison table sorted by mean return."""
    sub(title)
    if note:
        print(f"  {note}\n")
    print(f"  {'Variant':<30} {'Mean':>7} {'Excess':>8} {'IR':>6} {'Hit':>6} "
          f"{'Worst':>8} {'Turn':>6}")
    print(f"  {'-'*74}")
    for name, s in sorted(results.items(), key=lambda kv: -(kv[1]["mean"] if kv[1] else -9)):
        if not s:
            continue
        turn = f"{s['turnover']:.0%}" if not np.isnan(s["turnover"]) else "  -"
        print(f"  {name:<30} {s['mean']:>6.2%} {s['excess']:>+8.2%} "
              f"{s['ir']:>6.2f} {s['hit']:>5.0%} {s['worst_hold']:>7.2%} {turn:>6}")


# Market baseline
mkt_res, _ = run("mom_6m")          # any signal — market column is the same
MARKET = float(mkt_res["market"].mean())
print(f"  Market baseline: {MARKET:.2%} mean 6M forward return\n")
report["market"] = round(MARKET, 4)

# ════════════════════════════════════════════════════════════════════════════════
# TEST 1 — HORIZON
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 1 — WHICH HORIZON?")

horizons = {
    "mom_3m  (3 month)":        "mom_3m",
    "mom_6m  (6 month)":        "mom_6m",
    "mom_12m (12 month)":       "mom_12m",
    "mom_12_2 (12m skip 1m)":   "mom_12_2",
    "mom_avg (3+6+12 average)": "mom_avg",
}
r1 = {}
for label, col in horizons.items():
    res, t = run(col)
    r1[label] = stats(res, t)

table(r1, "Raw momentum value, top 20",
      "Higher IR means the excess return is more consistent, not just larger.")
report["horizons"] = r1

best_h_label = max((k for k in r1 if r1[k]), key=lambda k: r1[k]["mean"])
best_h = horizons[best_h_label]
print(f"\n  Best horizon: {best_h_label}")

# ════════════════════════════════════════════════════════════════════════════════
# TEST 2 — RANKING METHOD
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 2 — RANKING METHOD")

r2 = {}
for base in ["mom_6m", "mom_12_2"]:
    for suffix, label in [("", "raw value"),
                          ("_x", "cross-sectional pct"),
                          ("_sec", "sector-relative z")]:
        col = base + suffix
        if col not in bt.columns:
            continue
        res, t = run(col)
        r2[f"{base} — {label}"] = stats(res, t)

table(r2, "Same signal, different normalisation",
      "Cross-sectional ranks against all stocks; sector-relative against peers only.")
report["ranking"] = r2

# ════════════════════════════════════════════════════════════════════════════════
# TEST 3 — VOLATILITY ADJUSTMENT
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 3 — RISK-ADJUSTED MOMENTUM")

print("""
  Dividing momentum by volatility favours steady climbers over violent ones.
  Well documented in the literature as an improvement on raw momentum.
""")

r3 = {
    "mom_6m plain":            stats(*run("mom_6m")),
    "mom_6m / volatility":     stats(*run("mom_6m_vadj")),
    "mom_12_2 plain":          stats(*run("mom_12_2")),
    "mom_12_2 / volatility":   stats(*run("mom_12_2_vadj")),
    "blend: mom + vol-adj":    stats(*run("blend_vadj")),
}
table(r3, "Volatility adjustment")
report["risk_adjusted"] = r3

# ════════════════════════════════════════════════════════════════════════════════
# TEST 4 — BLENDS
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 4 — BLENDING HORIZONS")

r4 = {
    "mom_6m alone":            stats(*run("mom_6m")),
    "mom_12_2 alone":          stats(*run("mom_12_2")),
    "blend 6m + 12-2":         stats(*run("blend_6_122")),
    "blend 3m + 12m":          stats(*run("blend_3_12")),
    "mom_avg (3+6+12)":        stats(*run("mom_avg")),
}
table(r4, "Averaging cross-sectional ranks of two horizons")
report["blends"] = r4

# ════════════════════════════════════════════════════════════════════════════════
# TEST 5 — PORTFOLIO SIZE
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 5 — PORTFOLIO SIZE")

print("""
  Fewer names concentrates the signal but raises single-stock risk.
  More names dilutes toward the market.
""")

r5 = {}
for n in SIZES:
    res, t = run(best_h, top_n=n)
    r5[f"top {n} by {best_h}"] = stats(res, t)

table(r5, f"Portfolio size using {best_h}")
report["sizes"] = r5

# ════════════════════════════════════════════════════════════════════════════════
# TEST 6 — LONG-SHORT
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 6 — IS THE BOTTOM AS BAD AS THE TOP IS GOOD?")

top_res, _ = run(best_h, ascending=False)
bot_res, _ = run(best_h, ascending=True)

if top_res is not None and bot_res is not None:
    merged = top_res.merge(bot_res, on="date", suffixes=("_top", "_bot"))
    spread = merged["ret_top"] - merged["ret_bot"]
    print(f"""
  Top {TOP_N}      : {merged['ret_top'].mean():>7.2%}
  Bottom {TOP_N}   : {merged['ret_bot'].mean():>7.2%}
  Market       : {merged['market_top'].mean():>7.2%}
  Long-short   : {spread.mean():>7.2%}
  Spread > 0   : {(spread > 0).mean():>7.0%} of snapshots

  A wide spread means the signal ranks the whole universe well, not just
  the winners. If the bottom is close to the market, the edge is one-sided.""")
    report["long_short"] = {
        "top": round(float(merged["ret_top"].mean()), 4),
        "bottom": round(float(merged["ret_bot"].mean()), 4),
        "spread": round(float(spread.mean()), 4),
        "spread_positive_pct": round(float((spread > 0).mean()), 4),
    }

# ════════════════════════════════════════════════════════════════════════════════
# WINNER — YEAR BY YEAR
# ════════════════════════════════════════════════════════════════════════════════

hr("BEST VARIANT — YEAR BY YEAR")

everything = {}
for d in (r1, r2, r3, r4):
    everything.update({k: v for k, v in d.items() if v})

winner_label = max(everything, key=lambda k: everything[k]["mean"])
print(f"\n  Highest mean return: {winner_label}  "
      f"({everything[winner_label]['mean']:.2%})")

best_ir_label = max(everything, key=lambda k: everything[k]["ir"]
                    if not np.isnan(everything[k]["ir"]) else -9)
print(f"  Highest consistency (IR): {best_ir_label}  "
      f"(IR {everything[best_ir_label]['ir']:.2f})")

# Resolve label back to a column for the year table
col_lookup = {}
col_lookup.update(horizons)
for base in ["mom_6m", "mom_12_2"]:
    col_lookup[f"{base} — raw value"] = base
    col_lookup[f"{base} — cross-sectional pct"] = base + "_x"
    col_lookup[f"{base} — sector-relative z"] = base + "_sec"
col_lookup.update({
    "mom_6m plain": "mom_6m", "mom_6m / volatility": "mom_6m_vadj",
    "mom_12_2 plain": "mom_12_2", "mom_12_2 / volatility": "mom_12_2_vadj",
    "blend: mom + vol-adj": "blend_vadj",
    "mom_6m alone": "mom_6m", "mom_12_2 alone": "mom_12_2",
    "blend 6m + 12-2": "blend_6_122", "blend 3m + 12m": "blend_3_12",
    "mom_avg (3+6+12)": "mom_avg",
})

win_col  = col_lookup.get(winner_label, "mom_6m")
win_res, _  = run(win_col)
base_res, _ = run("mom_6m")

if win_res is not None and base_res is not None:
    m = win_res.merge(base_res, on="date", suffixes=("_win", "_base"))
    m["year"] = m["date"].dt.year
    sub(f"{winner_label} vs plain mom_6m vs market")
    print(f"  {'Year':<7} {'Market':>9} {'mom_6m':>9} {'Winner':>9} {'Diff':>8}")
    print(f"  {'-'*46}")
    for year, g in m.groupby("year"):
        print(f"  {year:<7} {g['market_win'].mean():>8.1%} "
              f"{g['ret_base'].mean():>8.1%} {g['ret_win'].mean():>8.1%} "
              f"{g['ret_win'].mean()-g['ret_base'].mean():>+7.1%}")

report["winner"]     = winner_label
report["winner_col"] = win_col

# ════════════════════════════════════════════════════════════════════════════════
# CURRENT PICKS
# ════════════════════════════════════════════════════════════════════════════════

hr("CURRENT PICKS")

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()

sub(f"Top {TOP_N} by {win_col}  ({latest_date.date()})")
picks = latest.dropna(subset=[win_col]).nlargest(TOP_N, win_col)
print(f"  {'#':<4}{'Ticker':<8} {'Sector':<24} {'Mom6M':>8} {'Mom12-2':>9} "
      f"{'Vol':>7} {'Mkt cap':>9}")
print(f"  {'-'*72}")
for i, (_, r) in enumerate(picks.iterrows(), 1):
    mc = r.get("market_cap", np.nan)
    print(f"  {i:<4}{r['ticker']:<8} {str(r['sector'])[:23]:<24} "
          f"{r.get('mom_6m', 0)*100:>7.1f}% {r.get('mom_12_2', 0)*100:>8.1f}% "
          f"{r.get('vol_raw', 0)*100:>6.1f}% "
          f"{('$'+format(mc/1e9, '.1f')+'B') if pd.notna(mc) else 'n/a':>9}")

report["current_picks"] = picks["ticker"].tolist()

# Sector concentration — momentum portfolios often pile into one sector
sub("Sector concentration of current picks")
conc = picks["sector"].value_counts()
for s, n in conc.items():
    bar = "#" * n
    print(f"  {s:<26} {n:>2}  {bar}")
if conc.iloc[0] >= TOP_N * 0.4:
    print(f"\n  {conc.index[0]} is {conc.iloc[0]/TOP_N:.0%} of the portfolio.")
    print("  Momentum tends to concentrate. Consider a per-sector cap if that")
    print("  concentration is more risk than you want.")

# Comparison with the ML model
sub("Versus the current ML model")
try:
    live = db_read("SELECT ticker, outperform_prob FROM live_predictions "
                   "ORDER BY outperform_prob DESC")
    ml_top    = live.head(TOP_N)["ticker"].tolist()
    ml_strong = live[live["outperform_prob"] >= 0.65]["ticker"].tolist()
    overlap   = set(picks["ticker"]) & set(ml_top)
    print(f"  ML STRONG calls        : {', '.join(ml_strong) if ml_strong else 'none'}")
    print(f"  Overlap in top {TOP_N}      : {len(overlap)}  "
          f"{', '.join(sorted(overlap)) if overlap else ''}")
    print(f"  Momentum only          : "
          f"{', '.join(sorted(set(picks['ticker']) - set(ml_top)))}")
except Exception as e:
    print(f"  Could not load live_predictions ({e})")

# ════════════════════════════════════════════════════════════════════════════════
# TEST 8 — DOES THE STAGE 4 mom_6m_rank BUG MATTER?
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 8 — THE mom_6m_rank BUG")

print("""
  Stage 4 computes:  groupby("ticker")["mom_6m"].rank(pct=True)

  That ranks each stock against its OWN history — "is this stock's momentum
  high by its own standards". The intent was groupby("date"), ranking each
  stock against every OTHER stock on that date.

  Both are legitimate signals. This is which one actually predicts better.
""")

bt["_own_hist"] = bt.groupby("ticker")["mom_6m"].rank(pct=True)

r8 = {}
res, t = run("mom_6m")
r8["cross-sectional (intended)"] = stats(res, t)
res, t = run("_own_hist")
r8["own history (Stage 4 today)"] = stats(res, t)

table(r8, "Same underlying signal, different reference point")
report["rank_bug"] = r8

_a = r8["cross-sectional (intended)"]
_b = r8["own history (Stage 4 today)"]
if _a and _b:
    _d = _a["mean"] - _b["mean"]
    print(f"\n  Difference: {_d:+.2%}")
    if _d > 0.01:
        print("""
  ^ Cross-sectional wins. Fix Stage 4 — one word:
        prices.groupby("ticker")["mom_6m"]  ->  prices.groupby("date")["mom_6m"]
    then rerun stages 4 and 6.""")
    elif _d < -0.01:
        print("""
  ^ Own-history ranking is BETTER. The 'bug' is picking up a genuine
    mean-reversion-against-self signal. Leave Stage 4 alone and rename the
    column so the next person is not misled.""")
    else:
        print("""
  ^ Within noise. Fix it for clarity, but expect no change in results.""")

# ════════════════════════════════════════════════════════════════════════════════
# TEST 9 — NON-OVERLAPPING WINDOWS
# ════════════════════════════════════════════════════════════════════════════════

hr("TEST 9 — NON-OVERLAPPING WINDOWS  (the honest consistency check)")

print("""
  Every number above rebalances monthly while holding six months, so
  consecutive observations share five months of the same returns. Hit rates
  and information ratios computed that way are inflated — the observations
  are not independent.

  This samples every 6th month instead. Six possible starting offsets, each
  giving a clean non-overlapping series.
""")

# Run on plain mom_6m — the current production signal. If a different
# variant wins earlier, rerun this section against that column too.
_res, _ = run("mom_6m")

if _res is not None and len(_res):
    _res = _res.sort_values("date").reset_index(drop=True)
    _means, _excs, _hits = [], [], []
    print(f"  {'Offset':<9} {'n':>4} {'Mean':>8} {'Excess':>9} {'Hit':>6}")
    print(f"  {'-'*40}")
    for _o in range(6):
        _sl = _res.iloc[_o::6]
        if len(_sl) < 5:
            continue
        _e = (_sl["ret"] - _sl["market"]).mean()
        _h = (_sl["ret"] > _sl["market"]).mean()
        _means.append(_sl["ret"].mean())
        _excs.append(_e)
        _hits.append(_h)
        print(f"  {_o:<9} {len(_sl):>4} {_sl['ret'].mean():>7.2%} "
              f"{_e:>+8.2%} {_h:>5.0%}")

    if _excs:
        print(f"""
  Across all starting offsets:
    mean excess  {np.mean(_excs):>+6.2%}   range {min(_excs):+.2%} to {max(_excs):+.2%}
    mean hit     {np.mean(_hits):>6.0%}   range {min(_hits):.0%} to {max(_hits):.0%}
""")
        if min(_excs) > 0:
            print("  ^ Positive excess at every starting point. Not a calendar artefact.")
        else:
            print("""  ^ Excess turns negative at some starting points. The edge depends on
    when you begin — treat the headline hit rate as optimistic.""")

        report["nonoverlap_excess_mean"] = round(float(np.mean(_excs)), 4)
        report["nonoverlap_excess_min"]  = round(float(min(_excs)), 4)
        report["nonoverlap_hit_mean"]    = round(float(np.mean(_hits)), 4)

# ════════════════════════════════════════════════════════════════════════════════
# VERDICT
# ════════════════════════════════════════════════════════════════════════════════

hr("VERDICT")

w = everything[winner_label]
b = r1.get("mom_6m  (6 month)")

print(f"""
  Market baseline        {MARKET:>7.2%}
  Current (plain 6m)     {b['mean']:>7.2%}   IR {b['ir']:.2f}   hit {b['hit']:.0%}
  Best variant           {w['mean']:>7.2%}   IR {w['ir']:.2f}   hit {w['hit']:.0%}
    -> {winner_label}

  Improvement over current: {w['mean'] - b['mean']:+.2%}
""")

if w["mean"] - b["mean"] > 0.01:
    print(f"  Worth switching to {winner_label}.")
elif w["mean"] - b["mean"] > 0.003:
    print(f"  Modest gain. Check the year-by-year table — if it comes from one")
    print(f"  or two years, it is probably noise rather than a better signal.")
else:
    print("  No variant meaningfully beats plain 6-month momentum. That is a")
    print("  useful result: the simplest version is already the best one.")

print("""
  Two caveats on every number above:

  Transaction costs are not modelled. At the turnover rates shown, a few
  tenths of a percent per rebalance is realistic, which eats into the excess.

  Returns overlap. Each snapshot holds for 6 months while snapshots are
  monthly, so consecutive observations are not independent and the true
  standard errors are wider than they look.""")

with open("momentum_lab.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\n" + "=" * 76)
print("  Saved -> momentum_lab.json    Database unchanged.")
print("=" * 76 + "\n")
