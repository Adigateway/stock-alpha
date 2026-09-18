# Stock Alpha

A momentum screen for S&P 500 stocks, with the full record of what was tried
and what failed.

Ranks the universe by six-month price return and surfaces the top fifteen.
One feature, no model, no tuning. It arrived there by elimination.

---

## What it does

```
SEC EDGAR + Yahoo Finance  ->  PostgreSQL  ->  rank by 6M return  ->  web UI
```

Runs quarterly via cron. Fundamentals are stored and displayed for screening
but play no part in the ranking.

---

## What was tested and removed

The interesting part of this project is the negative results.

| Approach | Result |
|---|---|
| Gradient-boosted model, 38 features | AUC 0.50 under blind validation. Removed. |
| Fundamental hard gates | Gated universe returned 8.74% vs market 9.36% |
| Sector-relative quality ranking | 10.28% vs 11.56% for gates + momentum |
| Multi-failure veto | Vetoed stocks returned 24.86% vs the 13.82% that replaced them |
| SEC 8-K sentiment (VADER + TF-IDF) | Failed ablation, added noise |
| Macro regime features (FRED) | Reduced OOS AUC |
| Six momentum variants | None beat plain 6-month return |

Each was measured, not assumed. Scripts are in `research/`.

### Why the ML model went

Blind validation across five snapshots, training stopped six months before
each one:

```
momentum_only        AUC 0.4999   rank corr +0.007
current_production   AUC 0.5089   rank corr +0.019
full (38 features)   AUC 0.5034   rank corr -0.002
```

AUC 0.50 over roughly 1,375 stock-outcome pairs is a coin flip. The three
feature sets also picked almost entirely different stocks on the same date,
which is what fitting noise looks like.

An earlier walk-forward test showed 17.92%, but it trained on rows whose
six-month outcomes were not yet known. That was leakage, not edge.

### Why fundamentals failed

Consistent with the literature. Piotroski's F-Score, the canonical fundamental
screen, was tested by firm size in the original 1999 paper: small firms showed
a 27.0% annual spread, large firms 1.7% and not statistically significant.
The S&P 500 is the one segment where fundamental screening has no documented
edge — every name has dozens of analysts already covering it.

---

## Performance

Blind validation, five snapshots 2021-2025, top 15 held six months:

| | Return | vs universe |
|---|---|---|
| Momentum | 24.16% | +14.25% |
| Universe | 9.91% | — |

Beat the universe in 3 of 5. Per snapshot:

```
2021   -11.66%
2022    -2.26%
2023   +26.07%
2024    +8.60%
2025   +50.50%
```

Two of five negative, and 2025 carries most of the average — that was the AI
and semiconductor rally. Excluding it the edge is roughly +5%. No transaction
costs are included and turnover runs near 40% a month.

---

## Bugs found along the way

Most of the work was debugging, not modelling.

**SEC extraction.** A `break` in the tag loop meant only the first matching
XBRL tag was ever read. Large filers moved from `Revenues` to the ASC 606 tag
around 2018 but kept stale legacy entries, so the loop found the old tag and
stopped. Apple, Nvidia and JPMorgan had zero rows; Microsoft had 8 quarters
ending 2010. Fixing it took the dataset from 8,438 rows to 14,798.

**Silent universe deletion.** `shares_outstanding` was missing on half of
filings, making `market_cap` NaN, which the microcap filter then dropped.
Roughly 40% of the universe was being excluded by accident.

**Look-ahead in the merge.** The SEC `end` field is the quarter's period end,
not the filing date. Fundamentals were being read ~45 days before they were
public. Fixed with an explicit lag.

**Cross-sectional rank bug.** `mom_6m_rank` was grouped by ticker instead of
date, ranking each stock against its own history rather than the market. That
column was one of four surviving the ablation and was worth 4.4% a year once
corrected.

---

## Layout

```
pipeline/
  stock_alpha.py        8-stage data pipeline
  live_signals.py       momentum ranking, writes to Postgres
research/
  13_screen_diagnostic  data quality audit
  16_momentum_lab       nine momentum variants
  17_ml_vs_momentum     walk-forward head-to-head
  18_blind_validation   blind snapshots, no look-ahead
webapp/
  api.py                Flask API
  index.html            frontend, single file
scripts/
  run_pipeline.sh       quarterly cron runner
config/
  db_config.example.py  template — copy and fill in
data/
  ticker_sectors.csv    universe definition
```

---

## Setup

```bash
git clone <repo-url> && cd stock-alpha
pip install -r requirements.txt

cp config/db_config.example.py db_config.py
# edit db_config.py with your credentials
```

PostgreSQL:

```sql
CREATE DATABASE stock_alpha;
CREATE USER stockuser WITH ENCRYPTED PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE stock_alpha TO stockuser;
GRANT ALL ON SCHEMA public TO stockuser;
```

First run, in order. Stage 1 is slow — one SEC request per ticker.

```bash
python3 pipeline/stock_alpha.py --only 1   # SEC filings
python3 pipeline/stock_alpha.py --only 2   # prices
python3 pipeline/stock_alpha.py --only 3   # ratios
python3 pipeline/stock_alpha.py --only 4   # price features
python3 pipeline/stock_alpha.py --only 6   # model dataset
python3 pipeline/live_signals.py           # ranking
```

Stages 5, 7 and 8 exist but are unused — sentiment failed ablation, and 7 and 8
trained and backtested the removed ML model.

Quarterly automation:

```
0 2 1 1,4,7,10 * /path/to/scripts/run_pipeline.sh
```

Web app: Gunicorn on `api.py` behind Nginx, serving `index.html`.

---

## Notes

The pipeline paths are absolute (`/home/aditya/stock_alpha`) and would need
changing for another machine.

SEC EDGAR requires a real contact address in the User-Agent header — set
`USER_AGENT` in `stock_alpha.py` before running stage 1.

`yfinance` is unofficial and occasionally breaks with Yahoo's changes.

---

## Disclaimer

Personal research project. Not financial advice.

Momentum ranks by price, not business quality — the list regularly contains
companies that are losing money, and they are labelled as such. It also
concentrates heavily in whichever sector is running, which is the opposite of
diversification.

Historical results do not predict future performance, and this record covers a
single market regime with no prolonged bear market in it.
