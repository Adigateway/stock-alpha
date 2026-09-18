"""
api.py — Stock Alpha API (momentum)
════════════════════════════════════════════════════════════════════════════════
Serves the momentum ranking from live_signals.

Changed from the ML version:
  - reads live_signals, not live_predictions
  - returns mom_rank and mom_pct, not a probability
  - reasoning is generated from momentum rank and fundamentals, and says
    plainly when a company is losing money
  - /api/validation returns the blind-validation record so the frontend
    never hardcodes performance numbers

Run:  sudo systemctl restart stockalpha
"""

import datetime
import json
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, "/home/aditya/stock_alpha")
from db_config import db_read, db_write

app = Flask(__name__)
CORS(app)

TOP_N = 15

# Blind validation, 18_blind_validation.py — five snapshots, six-month
# training gap, no look-ahead. The only performance numbers we publish.
VALIDATION = {
    "method": "Blind historical validation, 5 snapshots 2021-2025",
    "note": ("Training stops six months before each snapshot, because a row "
             "dated T only has its outcome at T+6. Top 15 held six months."),
    "momentum_mean": 0.2416,
    "universe_mean": 0.0991,
    "excess": 0.1425,
    "beat_universe": "3 of 5",
    "per_snapshot": [
        {"year": 2021, "excess": -0.1166},
        {"year": 2022, "excess": -0.0226},
        {"year": 2023, "excess": 0.2607},
        {"year": 2024, "excess": 0.0860},
        {"year": 2025, "excess": 0.5050},
    ],
    "caveat": ("Two of five snapshots were negative and 2025 carries most of "
               "the average — that was the AI and semiconductor rally. "
               "Excluding it the edge is roughly +5%. Transaction costs are "
               "not included and turnover runs near 40% a month."),
    "ml_note": ("A gradient-boosted model on 38 features was tested on the "
                "same snapshots and scored AUC 0.50 with near-zero rank "
                "correlation — no better than chance. It was removed."),
}


# ══════════════════════════════════════════════════════════════════════════════
# REASONING — derived from the data, never invented
# ══════════════════════════════════════════════════════════════════════════════


def clean(d):
    """
    Replace NaN and Inf with None. Python's json writes bare NaN, which is
    invalid JSON, so the browser rejects the entire response body.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif v is pd.NaT:
            out[k] = None
        else:
            out[k] = v
    return out


def safe_float(v, default=0.0):
    """float() that survives None and NaN."""
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def make_reasons(row, universe_n):
    r = []
    rank = int(row.get("mom_rank") or 0)
    mom  = safe_float(row.get("mom_6m_true"))
    roa  = safe_float(row.get("roa"))
    rev  = safe_float(row.get("revenue_yoy"))
    opm  = safe_float(row.get("operating_margin"))
    cfy  = safe_float(row.get("cf_yield"))
    mcap = safe_float(row.get("market_cap"))
    sector = str(row.get("sector") or "")

    # The signal itself
    if rank:
        r.append(f"Ranked #{rank} of {universe_n} by 6-month price return "
                 f"({mom*100:+.1f}%). This is the entire signal.")

    # Momentum character
    if mom > 1.0:
        r.append("More than doubled in six months — a large move already "
                 "happened, so entry price carries real reversal risk.")
    elif mom > 0.4:
        r.append("Strong sustained move over six months.")
    elif mom > 0.15:
        r.append("Moderate upward trend.")

    # Profitability — stated plainly
    if roa < 0:
        r.append(f"LOSING MONEY: return on assets {roa*100:.1f}%. Momentum "
                 f"ranks on price, not profitability. Verify before acting.")
    elif roa < 0.03:
        r.append(f"Thin profitability: ROA {roa*100:.1f}%. Marginal, not broken.")
    else:
        r.append(f"Solidly profitable: ROA {roa*100:.1f}%.")

    # Growth
    if rev > 0.5:
        r.append(f"Revenue up {rev*100:.0f}% year over year — the price move "
                 f"is backed by real demand, not just a re-rating.")
    elif rev > 0.15:
        r.append(f"Revenue growing {rev*100:.0f}% year over year.")
    elif rev < -0.05:
        r.append(f"Revenue declining {rev*100:.0f}% year over year.")

    # Margin quality
    if rev > 0.4 and 0 < opm < 0.10:
        r.append("High revenue growth on a thin operating margin — volume-led "
                 "and cyclical rather than compounding.")
    elif opm > 0.25:
        r.append(f"Strong operating margin {opm*100:.0f}%.")

    # Cash yield, with the financials caveat
    if cfy > 0.15:
        if sector in ("Financial Services", "Real Estate"):
            r.append(f"Cash-flow yield reads {cfy*100:.0f}%, but for a "
                     f"{sector.lower()} firm this includes float and is not "
                     f"cash available to shareholders. Ignore it.")
        else:
            r.append(f"High cash-flow yield {cfy*100:.0f}%.")
    elif cfy < 0:
        r.append("Negative operating cash flow — the business is consuming "
                 "cash, not generating it.")

    if mcap > 5e11:
        r.append(f"Mega-cap (${mcap/1e9:.0f}B) — deep liquidity.")

    return r


def signal_of(row):
    rank = int(row.get("mom_rank") or 999)
    pct  = float(row.get("mom_pct") or 0)
    if rank <= TOP_N:
        return "BUY"
    return "UPPER" if pct >= 0.5 else "LOWER"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predictions")
def predictions():
    try:
        df = db_read("SELECT * FROM live_signals ORDER BY mom_rank")
        n = len(df)
        out = []
        for _, row in df.iterrows():
            d = clean(row.to_dict())
            d["signal"]  = signal_of(d)
            d["reasons"] = make_reasons(d, n)
            # profitability flag the frontend can colour on
            roa = safe_float(d.get("roa"))
            d["profitable"] = ("yes" if roa > 0.03
                               else "marginal" if roa > 0 else "no")
            out.append(d)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/<ticker>")
def prediction_one(ticker):
    try:
        df = db_read("SELECT * FROM live_signals")
        n = len(df)
        hit = df[df["ticker"].str.upper() == ticker.upper()]
        if hit.empty:
            return jsonify({"error": "not found"}), 404
        d = clean(hit.iloc[0].to_dict())
        d["signal"]  = signal_of(d)
        d["reasons"] = make_reasons(d, n)
        return jsonify(d)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/validation")
def validation():
    """Performance record. The frontend reads this instead of hardcoding."""
    return jsonify(VALIDATION)


@app.route("/api/runs")
def runs():
    try:
        df = db_read("""
            SELECT run_date, ticker, sector, mom_6m_true, mom_rank,
                   signal, roa, revenue_yoy, market_cap
            FROM prediction_runs
            ORDER BY run_date DESC, mom_rank ASC
        """)
        grouped = {}
        for _, row in df.iterrows():
            rd = str(row.get("run_date", ""))[:10]
            grouped.setdefault(rd, []).append(clean(row.to_dict()))

        out = []
        for rd, stocks in sorted(grouped.items(), reverse=True):
            buys = [s for s in stocks if s.get("signal") == "BUY"]
            out.append({
                "run_date": rd,
                "n_total":  len(stocks),
                "n_buy":    len(buys),
                "stocks":   stocks[:60],   # cap payload
                "buys":     buys,
            })
        return jsonify(out)
    except Exception as e:
        return jsonify([])


@app.route("/api/sectors")
def sectors():
    """Sector mix of the current top N — the concentration warning."""
    try:
        df = db_read(f"SELECT sector FROM live_signals WHERE mom_rank <= {TOP_N}")
        counts = df["sector"].value_counts().to_dict()
        total = sum(counts.values()) or 1
        return jsonify({
            "counts": counts,
            "top_sector": max(counts, key=counts.get) if counts else None,
            "top_share": max(counts.values()) / total if counts else 0,
        })
    except Exception as e:
        return jsonify({"counts": {}, "error": str(e)})


# ── Pipeline runner ───────────────────────────────────────────────────────────
pipeline_log, pipeline_running = [], False


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    global pipeline_running, pipeline_log
    if pipeline_running:
        return jsonify({"status": "already_running"})
    pipeline_running = True
    pipeline_log = ["Starting momentum refresh..."]

    def _run():
        global pipeline_running, pipeline_log
        try:
            proc = subprocess.Popen(
                ["python3", "/home/aditya/stock_alpha/live_signals.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                cwd="/home/aditya/stock_alpha")
            for line in proc.stdout:
                pipeline_log.append(line.rstrip())
            proc.wait()
            pipeline_log.append("Done.")
        except Exception as e:
            pipeline_log.append(f"ERROR: {e}")
        finally:
            pipeline_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/run/status")
def run_status():
    return jsonify({"running": pipeline_running, "log": pipeline_log[-120:]})


@app.route("/api/portfolio", methods=["GET", "POST"])
def portfolio():
    if request.method == "POST":
        data = request.json or {}
        df = pd.DataFrame([{
            "username": "aditya",
            "portfolio_json": json.dumps(data.get("portfolio", {})),
            "updated_at": datetime.datetime.now().isoformat(),
        }])
        db_write(df, "portfolio_aditya", if_exists="replace")
        return jsonify({"ok": True})
    try:
        df = db_read('SELECT portfolio_json FROM portfolio_aditya LIMIT 1')
        if not df.empty:
            return jsonify(json.loads(df.iloc[0]["portfolio_json"]))
    except Exception:
        pass
    return jsonify({})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)