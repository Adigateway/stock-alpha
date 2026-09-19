"""
cleanup_db.py — one-off: drop tables and files left over from the ML pipeline.

Nothing in stock_alpha.py, live_signals.py or api.py reads these any more.
Run AFTER the slimmed pipeline has completed once, so `fundamentals` exists:

  python3 stock_alpha.py && python3 live_signals.py
  python3 cleanup_db.py          # dry run — lists what would be removed
  python3 cleanup_db.py --yes    # actually drop / delete

Kept: ticker_sectors, financials, prices, fundamentals, live_signals,
      prediction_runs, portfolio_aditya
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import inspect, text

BASE = Path("/home/aditya/stock_alpha")   # where the pipeline writes its files
sys.path.insert(0, str(BASE))
from db_config import get_engine

OBSOLETE_TABLES = [
    "fundamental_factors",   # old stage 3 — full-history ratios
    "price_factors",         # old stage 4 — per-day momentum features
    "sec_8k_text_cache",     # old stage 5 — raw 8-K text (largest table)
    "sec_sentiment_raw",     # old stage 5
    "sentiment_features",    # old stage 5
    "model_dataset_clean",   # old stage 6 — replaced by `fundamentals`
    "ml_predictions",        # old stage 7
    "backtest_results",      # old stage 8
    "live_predictions",      # duplicate of live_signals; the API never read it
]

OBSOLETE_FILES = [
    "model.pkl", "cv_results.json", "model_diagnostics.png",
    "backtest_report.txt", "backtest_diagnostics.png",
]

ap = argparse.ArgumentParser()
ap.add_argument("--yes", action="store_true", help="actually drop / delete")
args = ap.parse_args()

engine   = get_engine()
existing = set(inspect(engine).get_table_names())

# Safety: the new pipeline must have produced its tables first
missing = {"fundamentals", "live_signals"} - existing
if missing:
    sys.exit(f"Refusing to clean up: {sorted(missing)} not found. "
             f"Run stock_alpha.py and live_signals.py first.")

tables = [t for t in OBSOLETE_TABLES if t in existing]
files  = [p for p in (BASE / f for f in OBSOLETE_FILES) if p.exists()]

print("Tables to drop :", ", ".join(tables) or "none")
print("Files to delete:", ", ".join(p.name for p in files) or "none")

if not args.yes:
    print("\nDry run. Re-run with --yes to apply.")
    sys.exit(0)

with engine.begin() as conn:
    for t in tables:
        conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        print(f"  dropped {t}")
for p in files:
    p.unlink()
    print(f"  deleted {p.name}")
print("Done.")
