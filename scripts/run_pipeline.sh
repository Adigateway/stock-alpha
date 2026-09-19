#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_pipeline.sh — quarterly refresh, called by cron
#
# stock_alpha.py has three stages: SEC financials, Yahoo prices, and a
# one-row-per-ticker fundamentals snapshot. live_signals.py then ranks.
#
# Stage 1 is the slowest — one SEC request per ticker with a rate limit.
#
# Cron entry (crontab -e):
#   0 2 1 1,4,7,10 * /home/aditya/stock_alpha/run_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -u
cd /home/aditya/stock_alpha || exit 1

LOGDIR=/home/aditya/stock_alpha/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"

exec >> "$LOG" 2>&1

echo "═══════════════════════════════════════════════"
echo "  Quarterly refresh — $(date)"
echo "═══════════════════════════════════════════════"

FAILED=0

stage () {
  local label="$1"; shift
  echo ""
  echo "── $label ─────────────────────────────────────"
  local t0=$SECONDS
  if "$@"; then
    echo "   OK  ($((SECONDS - t0))s)"
  else
    echo "   FAILED (exit $?) after $((SECONDS - t0))s"
    FAILED=1
  fi
}

stage "Stage 1 — SEC financials"       python3 stock_alpha.py --only 1
stage "Stage 2 — Yahoo prices"         python3 stock_alpha.py --only 2
stage "Stage 3 — fundamentals snapshot" python3 stock_alpha.py --only 3

# Only rank if the data rebuilt cleanly — a partial refresh would produce a
# ranking from stale or half-written tables.
if [ "$FAILED" -eq 0 ]; then
  stage "Momentum ranking"            python3 live_signals.py
else
  echo ""
  echo "!! A data stage failed. Skipping the ranking so the website keeps"
  echo "   serving the last good result rather than a partial one."
fi

# Keep the API warm — it caches nothing, but a restart clears any stuck state
sudo systemctl restart stockalpha 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════"
if [ "$FAILED" -eq 0 ]; then
  echo "  Finished clean — $(date)"
else
  echo "  Finished WITH FAILURES — $(date)"
  echo "  Check the stage output above."
fi
echo "═══════════════════════════════════════════════"

# Prune logs older than a year
find "$LOGDIR" -name 'run_*.log' -mtime +365 -delete 2>/dev/null || true
