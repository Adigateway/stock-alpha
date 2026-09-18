"""
13_screen_diagnostic.py
════════════════════════════════════════════════════════════════════════════════
SCREENING DIAGNOSTIC — read-only. Changes nothing. Predicts nothing.

Answers four questions before we build a quality screen:

  1. DATA VALIDITY   — how much of the fundamental data is actually usable?
  2. GATE SURVIVAL   — how many companies pass each hard gate, individually
                       and combined? Is the survival rate stable over time?
  3. SECTOR DEPTH    — which sectors are too thin for percentile ranking?
  4. VALUATION COVER — can we compute P/B and cash-flow yield from what we have?

Also measures FILING LAG — the gap between a quarter's period-end date and
the monthly snapshot it gets merged onto. This is the look-ahead bias check.

Reads: financials, fundamental_factors, model_dataset_clean, ticker_sectors
Writes: nothing to the DB. Prints a report. Saves screen_diagnostic.json.

Usage:
  python3 13_screen_diagnostic.py
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

# ════════════════════════════════════════════════════════════════════════════════
# GATE THRESHOLDS  ── the numbers we're testing for sanity
# ════════════════════════════════════════════════════════════════════════════════

GATES = {
    "operating_margin_positive": {
        "desc": "Operating margin > 0 (core business is profitable)",
        "col" : "operating_margin",
        "test": lambda s: s > 0,
    },
    "revenue_not_collapsing": {
        "desc": "Revenue YoY > -15% (business not shrinking sharply)",
        "col" : "revenue_yoy",
        "test": lambda s: s > -0.15,
    },
    "cashflow_positive": {
        "desc": "Operating cash flow > 0 (generating real cash)",
        "col" : "operating_cashflow",
        "test": lambda s: s > 0,
    },
    "roa_positive": {
        "desc": "ROA > 0 (assets are producing income)",
        "col" : "roa",
        "test": lambda s: s > 0,
    },
    "current_ratio_ok": {
        "desc": "Current ratio > 1.0 (can cover short-term obligations)",
        "col" : "current_ratio",
        "test": lambda s: s > 1.0,
    },
}

# Sectors where balance-sheet gates don't apply — leverage IS the business model
LEVERAGE_EXEMPT = {"Financial Services", "Real Estate", "Utilities"}

# Below this many names, percentile ranking within sector is mostly noise
THIN_SECTOR_MIN = 15


def hr(title=""):
    print("\n" + "=" * 74)
    if title:
        print(f"  {title}")
        print("=" * 74)


def sub(title):
    print(f"\n── {title} " + "─" * max(0, 68 - len(title)))


report = {}

# ════════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════════

hr("SCREENING DIAGNOSTIC — read-only, changes nothing")

print("\nLoading tables...")

fin    = db_read("SELECT * FROM financials")
fund   = db_read("SELECT * FROM fundamental_factors")
model  = db_read("SELECT * FROM model_dataset_clean")
sectors = db_read("SELECT * FROM ticker_sectors")

if "Sector" in sectors.columns:
    sectors = sectors.rename(columns={"Sector": "sector"})

fin["report_date"]  = pd.to_datetime(fin["report_date"])
fund["report_date"] = pd.to_datetime(fund["report_date"])
model["date"]       = pd.to_datetime(model["date"])

print(f"  financials          : {len(fin):,} rows  |  {fin['ticker'].nunique()} tickers")
print(f"  fundamental_factors : {len(fund):,} rows  |  {fund['ticker'].nunique()} tickers")
print(f"  model_dataset_clean : {len(model):,} rows  |  {model['ticker'].nunique()} tickers")
print(f"  ticker_sectors      : {len(sectors):,} rows")

# ════════════════════════════════════════════════════════════════════════════════
# 1. DATA VALIDITY  — is the math even usable?
# ════════════════════════════════════════════════════════════════════════════════

hr("1. DATA VALIDITY — broken math we must exclude before judging anything")

n_fund = len(fund)
validity = {}

def check(name, mask, note=""):
    n   = int(mask.sum())
    pct = n / n_fund if n_fund else 0
    validity[name] = {"n": n, "pct": round(pct, 4)}
    flag = "  <-- LOOK" if pct > 0.05 else ""
    print(f"  {name:<34} {n:>7,}  ({pct:>6.1%}){flag}")
    if note and pct > 0.05:
        print(f"      {note}")
    return mask

sub("Rows where a core ratio is meaningless")

if "total_equity" in fund.columns:
    m_negeq = check(
        "negative total_equity", fund["total_equity"] < 0,
        "ROE and debt_to_equity are inverted here — must exclude, not interpret."
    )
else:
    m_negeq = pd.Series(False, index=fund.index)
    print("  total_equity column MISSING from fundamental_factors")

if "revenue" in fund.columns:
    check("revenue <= 0 or missing",
          (fund["revenue"].isna()) | (fund["revenue"] <= 0),
          "operating_margin explodes when revenue is near zero.")

if "shares_outstanding" in fund.columns:
    m_shares = check(
        "shares_outstanding missing/zero",
        (fund["shares_outstanding"].isna()) | (fund["shares_outstanding"] <= 0),
        "market_cap and log_market_cap are wrong for these rows."
    )
else:
    print("  shares_outstanding column MISSING")

if "operating_cashflow" in fund.columns:
    check("operating_cashflow missing", fund["operating_cashflow"].isna(),
          "Cash-flow gate and CF yield cannot be computed for these.")
else:
    print("  operating_cashflow MISSING from fundamental_factors")

if "gross_profit" in fund.columns:
    check("gross_profit missing", fund["gross_profit"].isna(),
          "Gross margin unavailable — it is one of the better quality signals.")
else:
    print("  gross_profit MISSING from fundamental_factors")

report["validity"] = validity

# ── Market cap sanity on the merged panel ────────────────────────────────────
sub("market_cap sanity on model_dataset_clean")

if "market_cap" in model.columns:
    mc = model["market_cap"]
    implausible = (mc.isna()) | (mc <= 0) | (mc > 1e13)
    print(f"  rows with implausible market_cap  : {implausible.sum():,}  "
          f"({implausible.mean():.1%})")
    print(f"  median market_cap                 : ${mc.median()/1e9:,.1f}B")
    print(f"  p1 / p99                          : "
          f"${mc.quantile(0.01)/1e9:,.2f}B / ${mc.quantile(0.99)/1e9:,.1f}B")
    report["market_cap_implausible_pct"] = round(float(implausible.mean()), 4)
    if implausible.mean() > 0.05:
        print("  ^ log_market_cap is currently the model's top feature. "
              "This matters.")

# ── Staleness: how old is the newest filing per ticker? ──────────────────────
sub("Filing staleness (newest filing per ticker vs latest snapshot)")

latest_snap = model["date"].max()
newest_per_ticker = fund.groupby("ticker")["report_date"].max()
age_days = (latest_snap - newest_per_ticker).dt.days

print(f"  Latest monthly snapshot : {latest_snap.date()}")
print(f"  Median filing age       : {age_days.median():.0f} days")
print(f"  Tickers stale >180 days : {(age_days > 180).sum()}  "
      f"of {len(age_days)}")
print(f"  Tickers stale >365 days : {(age_days > 365).sum()}")
if (age_days > 365).sum() > 0:
    stale = age_days[age_days > 365].sort_values(ascending=False).head(10)
    print(f"  Worst offenders: {', '.join(stale.index[:10])}")

report["median_filing_age_days"] = int(age_days.median())
report["tickers_stale_180d"] = int((age_days > 180).sum())

# ════════════════════════════════════════════════════════════════════════════════
# 2. FILING LAG — the look-ahead bias check
# ════════════════════════════════════════════════════════════════════════════════

hr("2. FILING LAG — is the model seeing data before it was public?")

print("""
  Stage 1 stores the SEC 'end' field as report_date. That is the quarter's
  PERIOD END, not the filing date. A 10-Q for the quarter ending Mar 31 is
  typically filed 30-45 days later.

  Stage 6 merges fundamentals onto monthly snapshots with direction=backward,
  so a Mar 31 period-end attaches to the Apr 30 snapshot — potentially before
  the filing was public.
""")

# Reconstruct the merge to measure the gap
fund_s  = fund.sort_values("report_date")
model_s = model[["ticker", "date"]].sort_values("date")

merged_check = pd.merge_asof(
    model_s, fund_s[["ticker", "report_date"]].sort_values("report_date"),
    by="ticker", left_on="date", right_on="report_date", direction="backward",
)
merged_check = merged_check.dropna(subset=["report_date"])
gap = (merged_check["date"] - merged_check["report_date"]).dt.days

print(f"  Gap between period-end and the snapshot it is merged onto:")
print(f"    median : {gap.median():.0f} days")
print(f"    p10    : {gap.quantile(0.10):.0f} days")
print(f"    p25    : {gap.quantile(0.25):.0f} days")
print(f"\n  Rows where gap < 45 days (likely NOT yet public): "
      f"{(gap < 45).sum():,}  ({(gap < 45).mean():.1%})")

report["filing_gap_median_days"] = int(gap.median())
report["filing_gap_under_45d_pct"] = round(float((gap < 45).mean()), 4)

if (gap < 45).mean() > 0.10:
    print("""
  ^ This is real look-ahead bias. It barely matters today because
    fundamentals carry almost no weight. It will matter a lot once
    fundamentals become the primary gate.

    Fix: add ~45 days to report_date before the merge_asof in Stage 6.""")

# ════════════════════════════════════════════════════════════════════════════════
# 3. SECTOR DEPTH — which sectors are too thin to rank within?
# ════════════════════════════════════════════════════════════════════════════════

hr("3. SECTOR DEPTH — is percentile ranking within sector meaningful?")

latest = model[model["date"] == latest_snap].merge(
    sectors[["ticker", "sector"]], on="ticker", how="left"
).dropna(subset=["sector"])

sec_counts = latest["sector"].value_counts().sort_values(ascending=False)

print(f"\n  Ticker count per sector on {latest_snap.date()}:\n")
print(f"  {'Sector':<28} {'Count':>6}   {'Rank quality':<20}")
print(f"  {'-'*60}")

thin_sectors = []
for sector, n in sec_counts.items():
    if n >= 25:
        quality = "good"
    elif n >= THIN_SECTOR_MIN:
        quality = "usable"
    else:
        quality = "TOO THIN"
        thin_sectors.append(sector)
    print(f"  {sector:<28} {n:>6}   {quality:<20}")

print(f"\n  Sectors below {THIN_SECTOR_MIN} names: "
      f"{len(thin_sectors)}  {thin_sectors if thin_sectors else ''}")
if thin_sectors:
    n_affected = int(sec_counts[thin_sectors].sum())
    print(f"  Tickers affected: {n_affected} of {len(latest)}  "
          f"({n_affected/len(latest):.0%})")
    print("""
  For these, a within-sector percentile is coarse — one company moving
  shifts a rank by 10+ points. Options: merge into a neighbouring sector,
  or fall back to full-universe ranking for these names.""")

report["sector_counts"] = {k: int(v) for k, v in sec_counts.items()}
report["thin_sectors"]  = thin_sectors

# ════════════════════════════════════════════════════════════════════════════════
# 4. GATE SURVIVAL — individually, then combined
# ════════════════════════════════════════════════════════════════════════════════

hr("4. GATE SURVIVAL — how many companies pass each hard gate?")

# Build a working frame: latest snapshot + the raw fields the gates need
gate_df = latest.copy()

# Pull operating_cashflow / total_equity / gross_profit from the newest filing
extra_cols = [c for c in ["operating_cashflow", "total_equity",
                          "gross_profit", "revenue", "net_income"]
              if c in fund.columns]
if extra_cols:
    newest_fund = (
        fund.sort_values("report_date")
            .groupby("ticker")
            .last()
            .reset_index()[["ticker"] + extra_cols]
    )
    gate_df = gate_df.merge(newest_fund, on="ticker", how="left",
                            suffixes=("", "_fin"))

sub("Each gate in isolation")
print(f"  {'Gate':<30} {'Pass':>6} {'Fail':>6} {'N/A':>6}  {'Pass %':>7}")
print(f"  {'-'*62}")

gate_results = {}
gate_masks   = {}

for name, spec in GATES.items():
    col = spec["col"]
    if col not in gate_df.columns:
        print(f"  {name:<30} {'-':>6} {'-':>6} {'-':>6}  COLUMN MISSING")
        gate_results[name] = {"available": False}
        continue

    s     = gate_df[col]
    na    = s.isna()
    passm = spec["test"](s) & ~na
    failm = ~passm & ~na

    gate_masks[name] = passm | na      # treat missing as "not excluded"
    n_pass, n_fail, n_na = int(passm.sum()), int(failm.sum()), int(na.sum())
    pass_pct = n_pass / max(1, n_pass + n_fail)

    gate_results[name] = {
        "available": True, "pass": n_pass, "fail": n_fail,
        "na": n_na, "pass_pct": round(pass_pct, 4),
        "desc": spec["desc"],
    }
    print(f"  {name:<30} {n_pass:>6} {n_fail:>6} {n_na:>6}  {pass_pct:>6.1%}")

print(f"\n  Gate definitions:")
for name, spec in GATES.items():
    print(f"    {name:<30} {spec['desc']}")

report["gates_individual"] = gate_results

# ── Combined survival ────────────────────────────────────────────────────────
sub("Combined — all gates applied together")

if gate_masks:
    combined = pd.Series(True, index=gate_df.index)
    for m in gate_masks.values():
        combined &= m

    n_survive = int(combined.sum())
    print(f"  Universe before screen : {len(gate_df)}")
    print(f"  Survivors              : {n_survive}  "
          f"({n_survive/len(gate_df):.1%})")
    print(f"  Excluded               : {len(gate_df)-n_survive}")

    report["combined_survival_pct"] = round(n_survive / len(gate_df), 4)
    report["combined_survivors"]    = n_survive

    # Per-sector survival
    sub("Survival by sector — watch for sectors wiped out")
    print(f"  {'Sector':<28} {'Before':>7} {'After':>6}  {'Survive %':>10}")
    print(f"  {'-'*56}")

    gate_df["_survives"] = combined
    sector_surv = {}
    for sector in sec_counts.index:
        sub_df = gate_df[gate_df["sector"] == sector]
        before = len(sub_df)
        after  = int(sub_df["_survives"].sum())
        pct    = after / max(1, before)
        sector_surv[sector] = {"before": before, "after": after,
                               "pct": round(pct, 4)}
        flag = "  <-- WIPED OUT" if pct < 0.20 else ""
        print(f"  {sector:<28} {before:>7} {after:>6}  {pct:>9.1%}{flag}")

    report["sector_survival"] = sector_surv

    wiped = [s for s, v in sector_surv.items() if v["pct"] < 0.20]
    if wiped:
        print(f"""
  ^ {', '.join(wiped)} nearly eliminated.
    If these are Financials / Real Estate / Utilities, the gates are wrong
    for them — leverage and thin ROA are the business model there, not a
    warning sign. Exempt those sectors from the balance-sheet gates.""")

    # Which single gate is doing the most damage?
    sub("Marginal impact — survivors if each gate is REMOVED")
    print(f"  {'Gate removed':<30} {'Survivors':>10}  {'Change':>8}")
    print(f"  {'-'*52}")
    for name in gate_masks:
        others = pd.Series(True, index=gate_df.index)
        for k, m in gate_masks.items():
            if k != name:
                others &= m
        n_without = int(others.sum())
        print(f"  {name:<30} {n_without:>10}  {n_without-n_survive:>+8}")
    print("\n  A gate with a large change is doing most of the filtering.")

# ── Stability over time ──────────────────────────────────────────────────────
sub("Survival stability — month by month over the last 3 years")

hist = model.merge(sectors[["ticker", "sector"]], on="ticker", how="left")
hist = hist.dropna(subset=["sector"])
hist = hist[hist["date"] >= (latest_snap - pd.DateOffset(years=3))]

# Only gates whose columns exist in model_dataset_clean
hist_gates = {n: s for n, s in GATES.items() if s["col"] in hist.columns}

if hist_gates:
    hmask = pd.Series(True, index=hist.index)
    for spec in hist_gates.values():
        s = hist[spec["col"]]
        hmask &= (spec["test"](s) | s.isna())
    hist["_survives"] = hmask

    monthly = hist.groupby("date").agg(
        total=("ticker", "count"),
        survivors=("_survives", "sum"),
    )
    monthly["pct"] = monthly["survivors"] / monthly["total"]

    print(f"  Using {len(hist_gates)} of {len(GATES)} gates "
          f"(others need columns not in model_dataset_clean)")
    print(f"\n  Mean survival   : {monthly['pct'].mean():.1%}")
    print(f"  Std dev         : {monthly['pct'].std():.1%}")
    print(f"  Min / Max       : {monthly['pct'].min():.1%} / {monthly['pct'].max():.1%}")

    print(f"\n  Last 12 months:")
    for d, row in monthly.tail(12).iterrows():
        bar = "#" * int(row["pct"] * 40)
        print(f"    {str(d.date()):<12} {int(row['survivors']):>4}/{int(row['total']):>4}  "
              f"{row['pct']:>6.1%}  {bar}")

    report["survival_mean"] = round(float(monthly["pct"].mean()), 4)
    report["survival_std"]  = round(float(monthly["pct"].std()), 4)

    if monthly["pct"].std() > 0.10:
        print("""
  ^ Survival swings more than 10 points month to month. The gates are
    picking up noise, not a stable quality signal. Loosen them.""")
    elif monthly["pct"].mean() < 0.25:
        print("""
  ^ Fewer than a quarter of names survive. That is a very small candidate
    pool — diversification will suffer. Consider loosening.""")
    elif monthly["pct"].mean() > 0.80:
        print("""
  ^ Over 80% survive. The gates are barely filtering anything — they will
    not change your predictions much. Consider tightening.""")
    else:
        print("\n  ^ Survival rate looks stable and reasonable.")

# ════════════════════════════════════════════════════════════════════════════════
# 5. VALUATION COVERAGE — can we compute P/B and CF yield?
# ════════════════════════════════════════════════════════════════════════════════

hr("5. VALUATION COVERAGE — the metrics currently missing entirely")

print("""
  The model has no idea whether a stock is cheap or expensive. Two metrics
  are computable from fields already in the database:

    price_to_book = market_cap / total_equity
    cf_yield      = operating_cashflow / market_cap
""")

sub("Are the inputs present?")

have_mc = "market_cap" in gate_df.columns
have_eq = "total_equity" in gate_df.columns
have_cf = "operating_cashflow" in gate_df.columns

print(f"  market_cap in model_dataset_clean : {'YES' if have_mc else 'NO'}")
print(f"  total_equity available            : {'YES' if have_eq else 'NO'}")
print(f"  operating_cashflow available      : {'YES' if have_cf else 'NO'}")

val_report = {}

if have_mc and have_eq:
    eq = gate_df["total_equity"]
    mc = gate_df["market_cap"]
    computable = (eq > 0) & (mc > 0) & eq.notna() & mc.notna()
    pb = (mc / eq).where(computable)
    print(f"\n  price_to_book computable for {int(computable.sum())} "
          f"of {len(gate_df)} ({computable.mean():.0%})")
    print(f"    median : {pb.median():.2f}")
    print(f"    p10/p90: {pb.quantile(0.10):.2f} / {pb.quantile(0.90):.2f}")
    print(f"    (negative-equity names excluded — P/B is meaningless there)")
    val_report["pb_coverage"] = round(float(computable.mean()), 4)
    val_report["pb_median"]   = round(float(pb.median()), 3)

if have_mc and have_cf:
    cf = gate_df["operating_cashflow"]
    mc = gate_df["market_cap"]
    computable = (mc > 0) & cf.notna() & mc.notna()
    cfy = (cf / mc).where(computable)
    print(f"\n  cf_yield computable for {int(computable.sum())} "
          f"of {len(gate_df)} ({computable.mean():.0%})")
    print(f"    median : {cfy.median():.2%}")
    print(f"    p10/p90: {cfy.quantile(0.10):.2%} / {cfy.quantile(0.90):.2%}")
    val_report["cfy_coverage"] = round(float(computable.mean()), 4)
    val_report["cfy_median"]   = round(float(cfy.median()), 4)

if "gross_profit" in gate_df.columns and "revenue" in gate_df.columns:
    gp, rev = gate_df["gross_profit"], gate_df["revenue"]
    computable = (rev > 0) & gp.notna() & rev.notna()
    gm = (gp / rev).where(computable)
    print(f"\n  gross_margin computable for {int(computable.sum())} "
          f"of {len(gate_df)} ({computable.mean():.0%})")
    print(f"    median : {gm.median():.1%}")
    print(f"    (fetched in Stage 1, currently unused by the model)")
    val_report["gm_coverage"] = round(float(computable.mean()), 4)

report["valuation"] = val_report

print("""
  NOTE: total_equity, operating_cashflow and gross_profit are NOT in
  model_dataset_clean — Stage 6 KEEP_COLS drops them. To use these in a
  screen they must be added to KEEP_COLS and Stage 6 re-run.""")

# ════════════════════════════════════════════════════════════════════════════════
# VERDICT
# ════════════════════════════════════════════════════════════════════════════════

hr("VERDICT — what to fix before building the screen")

issues = []

if report.get("filing_gap_under_45d_pct", 0) > 0.10:
    issues.append(
        f"LOOK-AHEAD BIAS: {report['filing_gap_under_45d_pct']:.0%} of merges use "
        f"fundamentals that were likely not yet public. Add a 45-day lag in Stage 6."
    )

if report.get("market_cap_implausible_pct", 0) > 0.05:
    issues.append(
        f"MARKET CAP: {report['market_cap_implausible_pct']:.0%} of rows implausible. "
        f"log_market_cap is a top feature — this propagates."
    )

if report.get("thin_sectors"):
    issues.append(
        f"THIN SECTORS: {', '.join(report['thin_sectors'])} have <{THIN_SECTOR_MIN} "
        f"names. Within-sector percentiles will be noisy."
    )

wiped = [s for s, v in report.get("sector_survival", {}).items()
         if v["pct"] < 0.20]
if wiped:
    issues.append(
        f"SECTORS WIPED: {', '.join(wiped)} nearly eliminated by the gates. "
        f"Exempt leverage-heavy sectors from balance-sheet gates."
    )

sm = report.get("survival_mean")
if sm is not None:
    if sm < 0.25:
        issues.append(f"TOO TIGHT: only {sm:.0%} survive on average. Loosen.")
    elif sm > 0.80:
        issues.append(f"TOO LOOSE: {sm:.0%} survive — barely filtering. Tighten.")

ss = report.get("survival_std")
if ss is not None and ss > 0.10:
    issues.append(f"UNSTABLE: survival swings +/-{ss:.0%} month to month.")

if not val_report.get("pb_coverage"):
    issues.append("NO VALUATION: total_equity not in model_dataset_clean. "
                  "Add to Stage 6 KEEP_COLS to enable P/B.")

if issues:
    print()
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}\n")
else:
    print("\n  No blocking issues. Thresholds look sane — safe to build the screen.\n")

report["issues"] = issues

with open("screen_diagnostic.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("=" * 74)
print("  Saved -> screen_diagnostic.json")
print("  Nothing in the database was changed.")
print("=" * 74 + "\n")
