# Stock Alpha

A momentum screen for S&P 500 stocks. Every quarter it ranks the index by how
strongly each stock has been rising, relative to how bumpy the ride was, and
shows the top 15. A portfolio tab simulates what a basket of those stocks
might be worth in a year.

Personal research project. Not financial advice.

---

## The idea

**1. Collect only what's needed.** Daily closing prices for the last two years
(Yahoo Finance) and seven numbers per quarter from company filings (SEC EDGAR):
revenue, operating income, net income, assets, equity, operating cash flow and
shares outstanding.

**2. Filter.** Drop companies that are losing money (return on assets ≤ 0) or
are smaller than $300M in market value.

**3. Score.** For each remaining stock:

```
score = 6-month return (skipping the most recent month) ÷ annualized volatility
```

- *6-month return, skipping the latest month:* the price change from seven
  months ago to one month ago. The latest month is skipped because very
  short-term moves tend to reverse.
- *Annualized volatility:* how much the price swings day to day, measured over
  the last ~126 trading days and scaled to a year.

Dividing by volatility rewards steady climbers over stocks that got there by
lurching around.

**4. Rank.** Highest score first. The top 15 are marked BUY. Profitability,
growth and margins are shown next to each stock for your own judgement, but
only the filter in step 2 uses them.

**5. Simulate a portfolio.** Add stocks with **+** and the PORTFOLIO tab runs a
Monte Carlo simulation in the browser:

- It averages the stocks' yearly trend (μ) and volatility (σ).
- It plays out 1,000 possible years, day by day for 252 trading days. Each day
  the value moves by the trend plus a random shock sized by σ (Geometric
  Brownian Motion).
- It sorts the 1,000 end values and reports the **5th percentile** (bad case:
  95% of runs ended above it), the **median** (typical case) and the
  **95th percentile** (good case), for whatever starting amount you enter.

By default the trend is taken from each stock's recent 6-month return, which
assumes the recent run continues. That is optimistic. A toggle switches to a
flat 8% market average instead.

```
SEC EDGAR + Yahoo Finance → PostgreSQL → filter → score → rank → web app
                                                        portfolio → Monte Carlo
```

---

## How it got here

An earlier version was much bigger: a machine-learning model on 38 features,
news sentiment from company filings, and several fundamental screens. Each was
tested and removed.

| Tried | Why it was dropped |
|---|---|
| Machine-learning model (38 features) | Blind test scored AUC 0.50, the same as a coin flip |
| Fundamental screens (3 variants) | Each one lowered returns compared with momentum alone |
| Filing sentiment (VADER, TF-IDF) | Added noise, no signal |
| Macro-economic features | Made results worse |
| Other momentum variants | None beat the simple 6-month return |

What survived is plain 6-month momentum. In a blind test over five snapshots
(2021–2025, top 15 held six months) it returned **24.2% vs 9.9%** for the whole
index and beat it in 3 of 5. The two losing years were 2021 and 2022, and 2025
(the AI rally) carries most of the average. Without 2025 the edge is about
+5%. Transaction costs are not included.

These numbers are for plain momentum. The current version adds the
profitability filter and the volatility adjustment and has **not** been
blind-tested yet.

Bugs fixed along the way included the SEC parser reading only the first
matching tag (Apple, Nvidia and JPMorgan had no data), ~40% of stocks
silently dropped for missing share counts, and filings being used ~45 days
before they were public.

The scripts behind all of this are in `research/`. They read tables the
current pipeline no longer builds, so they are kept as a record, not to rerun.

---

## Running it

PostgreSQL setup:

```sql
CREATE DATABASE stock_alpha;
CREATE USER stockuser WITH ENCRYPTED PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE stock_alpha TO stockuser;
GRANT ALL ON SCHEMA public TO stockuser;
```

Copy `config/db_config.example.py` to `db_config.py` and fill it in. Set
`USER_AGENT` in `pipeline/stock_alpha.py` to a real email address (SEC
requires it). Then:

```bash
python3 pipeline/stock_alpha.py      # 1 filings, 2 prices, 3 fundamentals (stage 1 is slow)
python3 pipeline/live_signals.py     # filter, score, rank
```

Refresh quarterly with cron:

```
0 2 1 1,4,7,10 * /path/to/scripts/run_pipeline.sh
```

Web app: Gunicorn on `webapp/api.py` behind Nginx, serving `webapp/index.html`.

Upgrading a database built by the old version? Remove its leftover tables:

```bash
python3 scripts/cleanup_db.py          # shows what would be dropped
python3 scripts/cleanup_db.py --yes    # drops it
```

---

## Layout

```
pipeline/stock_alpha.py    collect data: filings, prices, fundamentals
pipeline/live_signals.py   filter, score and rank
webapp/api.py              Flask API
webapp/index.html          website, including the Monte Carlo
scripts/run_pipeline.sh    quarterly cron job
scripts/cleanup_db.py      one-off cleanup after upgrading
research/                  the old experiments
config/                    database config template
data/ticker_sectors.csv    the stock universe
```

Paths are hard-coded to `/home/aditya/stock_alpha`. `yfinance` is unofficial
and occasionally breaks.

---

## Disclaimer

Momentum ranks by price, not business quality, and tends to pile into
whichever sector is hot. The Monte Carlo is a spread of outcomes based on past
volatility, not a prediction. Past results cover one market period with no
long bear market. Don't invest based on this tool alone.
