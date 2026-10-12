# Crypto Quant Intelligence Platform (CQIP)

A research-grade crypto research backend: public market/news/on-chain data ingestion,
deep-learning return forecasting, reinforcement-learning trading, and leakage-free
walk-forward backtesting — exposed through a CLI and a FastAPI service.

Everything runs on **public, keyless data sources**. No exchange API keys are required.

---

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data sources](#data-sources)
- [CLI reference](#cli-reference)
- [HTTP API](#http-api)
- [Evaluating model quality](#evaluating-model-quality)
- [Configuration](#configuration)
- [Data layout](#data-layout)
- [Methodology notes](#methodology-notes)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## Architecture

```
  providers ──▶ ingest ──▶ Parquet (partitioned) ──▶ features ──▶ forecaster (DL)
  news_providers ──▶ sentiment ──┘                         │           │
  (RSS, F&G, on-chain)                                    │           ▼
                                                          └──▶ RL env (PPO)
                                                                       │
                          registry ◀── tracking ◀── backtest ◀─────────┘
                                    │
                                    ▼
                          FastAPI (REST + SSE + WebSocket) ──▶ background jobs
```

**Storage split.** Bulk time series go to partitioned Parquet (`year=/month=`); small,
mutable metadata — data coverage, experiments, runs, model versions, predictions,
backtest results, jobs — go to SQLite in WAL mode.

**Modules.**

| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` via pydantic-settings (`CQIP_*` env vars / `.env`) |
| `storage.py` | `ParquetStore`: hive-partitioned upserts, filters, atomic writes |
| `db.py` | SQLAlchemy models, `UTCDateTime` decorator, engine cache, coverage |
| `providers.py` | Exchange OHLCV providers (Gate, HTX, Bitget, …) with pagination |
| `news_providers.py` | RSS, SEC, Federal Reserve, mempool.space, CoinGecko, Alternative.me |
| `sentiment.py` | Transformer sentiment with a lexicon fallback |
| `features.py` | Causal technical, sentiment and on-chain features; targets and splits |
| `forecaster.py` | LSTM / Transformer / MLP probabilistic forecasters + artifacts |
| `rl_env.py` | Gymnasium single- and multi-asset trading environments |
| `backtest.py` | Causal execution, metrics, benchmarks, walk-forward validation |
| `pipeline.py` | Dataset building, training orchestration, persistence |
| `tracking.py` | Experiment tracking and model registry |
| `jobs.py` | Thread-pool job manager with DB-backed status |
| `api.py` | REST, SSE, WebSocket surface |
| `cli.py` | Typer command-line interface |

---

## Installation

Requires **Python 3.11+**.

```bash
py -m pip install -e .
```

Optional extras:

```bash
py -m pip install -e ".[dev]"   # pytest, ruff
```

> If the generated console script is not on your `PATH` (common on Windows), invoke the
> CLI as `py -m cqip.cli ...` instead of `cqip ...`.

Copy the environment template and adjust as needed:

```bash
cp .env.example .env
```

---

## Quick start

```bash
# 1. Verify storage and database connectivity
py -m cqip.cli check

# 2. Ingest 120 days of hourly candles, news and on-chain data
py -m cqip.cli ingest candles BTC_USDT ETH_USDT --interval 1h --lookback-days 120
py -m cqip.cli ingest news --lookback-days 120
py -m cqip.cli ingest onchain

# 3. Train a forecaster (CPU is fine, just slower)
py -m cqip.cli train forecaster BTC_USDT --interval 1h --kind lstm --epochs 30

# 4. Walk-forward backtest the registered model
py -m cqip.cli train backtest <model-id> --folds 3

# 5. Compare against baseline strategies
py -m cqip.cli train benchmark BTC_USDT --interval 1h

# 6. Start the API
py -m cqip.cli serve
```

Then open <http://127.0.0.1:8000/docs>.

---

## Data sources

All sources are public and unauthenticated.

**Market data** — Gate.io, HTX, Bitget (paginated REST klines). A Binance provider is
implemented and registered, but its API is unreachable from this environment.
**News** — CoinDesk, CoinTelegraph, Cointribune, Decrypt, Bitcoin Magazine,
The Block, regulatory (SEC) and macro (Federal Reserve) RSS.
**On-chain** — mempool.space (mempool fees, hashrate, difficulty, pool shares),
CoinGecko global metrics, Alternative.me Fear & Greed index.

CoinGecko is used through `/global` and `/market_chart`; its `/coins/markets` endpoint
returns HTTP 403 from here. Coinbase, Kraken, OKX, Bybit and Bitstamp are geo- or
network-restricted and have no working endpoint here. Because `version` lists registered
providers rather than verified reachability, treat a provider as usable only once
`ingest candles` has actually returned rows for it.

**Sentiment** uses `distilbert-base-uncased-finetuned-sst-2-english`. Set
`CQIP_ALLOW_MODEL_DOWNLOAD=false` to skip the download and use the built-in lexicon
fallback (lower quality, works offline).

---

## CLI reference

```
version                                Show environment and provider configuration
check                                  Verify database and storage connectivity
serve                                  Start the HTTP API server (--host, --port)

ingest candles SYMBOL...              --interval, --lookback-days, -p/--provider,
                                      --concurrency
ingest news                            --lookback-days
ingest onchain                         --no-fng
ingest all                             --interval, --lookback-days

train forecaster SYMBOL                --interval, --horizon, --kind, --epochs, --seq-len
train rl SYMBOL                        --interval, --timesteps, --n-envs
train backtest MODEL_ID                --threshold, --folds
train benchmark SYMBOL                 --interval

model list | show ID | predict ID | promote ID | remove ID
data info [DATASET]                    dataset info and coverage
data features SYMBOL                   --interval

research SYMBOL...                     --interval (repeatable), --model (repeatable),
                                      --epochs, --folds, --horizon, --threshold,
                                      --min-rows, --resume
```

Note: the candle provider flag is singular (`-p/--provider`, space- or comma-separated).
`train forecaster` accepts `--epochs` and `--seq-len` explicitly; every other forecaster
hyperparameter falls back to the `CQIP_FORECASTER_*` settings. `model predict` takes the
model id as a positional argument and no extra flags.

`model predict` and `model list` truncate long IDs in the table; `model show <id>` prints
the full identifier. Model IDs look like `btc-usdt-lstm-h1-0d595ebe`.

### Research sweeps

`research` runs a timeframe x architecture matrix in one command, resampling coarse
timeframes from the stored base series:

```bash
py -m cqip.cli research BTC_USDT ETH_USDT \
  --interval 1h --interval 4h --interval 1d \
  --model lstm --model transformer --model mlp \
  --epochs 5 --folds 3
```

Repeat `--interval` and `--model` rather than passing a comma-separated list. In
PowerShell an unquoted `1h,4h,1d` is parsed as arithmetic before Python sees it,
because `d` is a PowerShell numeric suffix and `1d` becomes the number `1`. A quoted
`--interval "1h,4h,1d"` also works.

Each cell is trained, walk-forward backtested, and judged against the quality gates
below. Cells with too little history are reported as `insufficient_data` rather than
failures, and a cell missing any required metric is never scored as passing. Reports
are written to `artifacts/reports/`, and `--resume` reuses completed cells from the
most recent sweep.

#### Popular-universe backfill

```bash
py -m cqip.cli universe                 # resolve the curated top-30 against Gate
py -m cqip.cli backfill --interval 1d --interval 1h
```

`universe` reports every requested ticker as tradable, a stablecoin, or unavailable,
naming known renames (`MATIC` -> `POL`) and any symbol the exchange does not list.
`backfill` stores the deepest history each provider will serve: roughly 20,000 days of
`1d` candles and one year of `1h`, because Gate stops paginating `1h` after ~9,000
bars. Daily depth reaches the earliest candle Gate has for a pair, which is not
necessarily the asset's original listing date.

#### Staged sweeps

A full matrix over 29 symbols is tens of hours on CPU, so `--staged` trains every cell
for one epoch, ranks the survivors by directional accuracy plus correlation, and
retrains only the top `--survivors` for the full `--epochs`:

```bash
py -m cqip.cli research BTC_USDT ETH_USDT ... --interval 1d \
  --model lstm --model mlp --staged --survivors 6 --horizon 12
```

Screening output is written to `sweep-screen-*.json` so it never overwrites the full
report that `sweep-latest.json` points at. Note that early stopping usually converges
within a few epochs, so a promoted cell can legitimately reproduce its screening
metrics exactly; identical rows across the two tables are not a caching bug.

#### Result on the popular universe (2026-09-30)

29 targets x `1d`/`1h` x `lstm`/`mlp` at horizon 12, 3 folds, staged: **116 cells
trained, 6 promoted, 0 passed the gates.**

The instructive failure mode is that daily cells show genuinely positive directional
accuracy and correlation (DOT 0.698 / +0.479, VET 0.628 / +0.488) while losing money
(Sharpe -5.73 and -1.72) on 11-13 trades. Directional accuracy is measured bar by bar
and ignores the size of each loss, so a model that is often right and occasionally
catastrophic clears an accuracy threshold and still fails. Median daily accuracy of
0.526 and a median of 7 trades per cell is the honest summary; only 12 `1h` cells
produced at least 30 trades and none of those reached 0.55 accuracy with positive
correlation.

Two bugs found while producing these numbers are now regression-tested: the
walk-forward backtest used to score every model at `horizon=1` regardless of the
horizon it was trained on, and the data-limit estimate demanded ~1,489 daily bars and
silently blocked `TIA`/`SUI`/`ARB`/`PEPE` (they train fine on ~1,100).

#### Full-universe backfill (2026-09-30)

`TOP_CRYPTO_SYMBOLS` holds 243 unique tickers. After deduplication (`HBAR`, `DYDX`
and `FET` each appeared twice) and resolution against Gate:

| group | count | notes |
| --- | --- | --- |
| prediction targets | 207 | tradable, non-stablecoin |
| stablecoins | 10 | pegged, so not directionally predictive |
| no venue found | 20 | see below |
| recoverable on another venue | 6 | `BGB`/`MX`/`ZEN`/`BEAM` on Bitget, `ICX`, `CAT` on HTX |

Stored: **2,244,742 candles across 216 symbols.**

| interval | rows | coverage | depth |
| --- | --- | --- | --- |
| `1d` | 381,802 | 2013-03-31 to present | median 1,961 days; `BTC_USDT` 4,931 |
| `1h` | 1,858,562 | 2025-09-29 to present | one year (exchange retention) |

118 of 216 symbols carry at least five years of daily history. `MKR` resolves to
`SKY_USDT` through the rename map. The 20 symbols no venue lists are the quote
asset itself (`USDT`), assets that never had a spot pair here (`TON`, `HYPE`, `LEO`,
`TAO`, `KCS`, `PI`, `MX`), delisted majors (`XMR`, `EOS`, `AGIX`, `OCEAN`), and
stablecoins with no Gate pair (`DAI`, `EURC`, `USDD`, `USTB`). Each is annotated
with its reason in `universe` output rather than reported as a bare gap.

#### News coverage is the hard limit

`ingest news` currently stores **334 articles spanning 64 days** (2026-07-27 to
present) from 12 RSS sources, all sentiment-scored. RSS only ever serves a feed's
current window, so this cannot be extended backwards without a provider that keeps
history. Inception-to-present news for these assets needs a paid or archive-backed
source, and no such integration exists in this repo yet.

---

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Status, database check, active providers, sentiment backend |
| `GET` | `/health/datasets` | Per-dataset row counts, columns, time range, size |
| `GET` | `/data/candles` | OHLCV series (`symbol`, `interval`, `limit`, `exchange`) |
| `GET` | `/data/symbols` | Ingested symbols |
| `GET` | `/data/news` | Scored articles (`source`, `category`, `limit`) |
| `GET` | `/data/coverage` | Recorded provider coverage ranges |
| `POST` | `/data/ingest` | Queue ingestion (202, runs in the background) |
| `GET` | `/models` | List models (`kind`, `symbol`, `limit`) |
| `POST` | `/models/train` | Queue forecaster training (202) |
| `POST` | `/models/train-rl` | Queue RL training (202) |
| `POST` | `/models/backtest` | Queue a walk-forward backtest (202) |
| `GET` | `/models/{id}` | Model detail |
| `GET` | `/models/{id}/predict` | Next-bar distribution |
| `GET` | `/models/{id}/forecast` | Forecast path (`limit`) |
| `POST` | `/models/{id}/promote` | Change registry stage |
| `DELETE` | `/models/{id}` | Unregister and delete the artifact |
| `POST` | `/jobs` | Submit a job: `{"kind": ..., "params": {...}}` |
| `GET` | `/jobs` | List jobs (`kind`, `status`, `limit`) |
| `GET` | `/jobs/{id}` | Job status and result |
| `DELETE` | `/jobs/{id}` | Cancel a job |
| `GET` | `/jobs/{id}/stream` | SSE progress stream |
| `GET` | `/training/runs` | Experiment runs |
| `WS` | `/stream/prices` | Polling price ticks (5s) |
| `WS` | `/stream/jobs/{id}` | Live job status and progress |

Interactive docs are served at `/docs`, the schema at `/openapi.json`.

---

## Configuration

Every setting is an environment variable prefixed with `CQIP_`, or a key in `.env`.
See [`.env.example`](.env.example) for the annotated full list. The most useful ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CQIP_DATA_DIR` | `./data` | Root for Parquet, artifacts, SQLite and cache |
| `CQIP_LOG_LEVEL` | `INFO` | Log verbosity |
| `CQIP_ALLOW_MODEL_DOWNLOAD` | `true` | `false` forces the lexicon sentiment fallback |
| `CQIP_FORECASTER_KIND` | `lstm` | `lstm`, `transformer` or `mlp` |
| `CQIP_BACKTEST_FEE_BPS` | `10` | Taker fee in basis points |
| `CQIP_BACKTEST_SLIPPAGE_BPS` | `5` | Slippage in basis points |
| `CQIP_BACKTEST_ANCHOR` | `expanding` | `expanding` or `rolling` walk-forward windows |
| `CQIP_CORS_ORIGINS` | `["*"]` | **Restrict this outside local development** |

---

## Data layout

```
data/
├── cqip.db                       SQLite: coverage, runs, models, predictions, jobs
├── parquet/
│   ├── ohlcv/year=2026/month=05/part-0.parquet
│   ├── news/year=2026/part-0.parquet
│   ├── onchain/year=2026/part-0.parquet
│   └── fear_greed/year=2018/.../year=2026/
├── artifacts/
│   ├── models/btc-usdt-lstm-h1-0d595ebe.pt
│   └── runs/
└── cache/
```

Datasets are keyed for upserts: `ohlcv` on `(ts, symbol, exchange)`, `news` on `uid`,
`onchain` and `fear_greed` on `(ts, asset, metric)`. Incremental writes rewrite the merged
dataset, so a `write` reports only genuinely **new** keys.

---

## Evaluating model quality

A green test suite says the code runs; it says nothing about whether the model is any
good. Check these in order, and stop at the first failure — later stages are meaningless
once an earlier one has collapsed.

**1. Forecast quality, before any P&L.** `model show <id>` reports the held-out
metrics. The three that decide whether a model has any edge:

| Metric | Passes when | Meaning |
| --- | --- | --- |
| `test_directional_accuracy` | `> 0.55` | Below `0.5` it is worse than a coin flip |
| `test_correlation` | `> 0.05` | Sign agreement at chance has no trading value |
| `test_68pct_interval_coverage` | `0.60`–`0.75` | Far below `0.68` means overconfident bands |

Accuracy must also beat the **base rate** of up-moves in the sample, which is not 50%
on a falling market. A constant "always short" signal scores exactly the base rate.

**2. Prediction distribution.** A collapsed model is the most common failure and is
invisible in a single accuracy number. Check that predictions actually vary and are
not one constant sign:

```python
pred = pipeline.trainer.predict(model, bundle.frame, feature_schema, model_info)
r = pred["predicted_return"].to_numpy()
print(np.percentile(r, [1, 25, 50, 75, 99]))   # should straddle zero
print((r > 0).mean())                            # should not be ~0 or ~1
```

If the median is negative and the 95th percentile is still negative, the network has
learned a constant bias rather than a conditional forecast. Re-train before trusting
anything downstream.

**3. Profitability against a benchmark.** Never read a strategy's return alone.

```bash
py -m cqip.cli train benchmark BTC_USDT --interval 1h
py -m cqip.cli train backtest <model-id> --folds 3
```

Compare the strategy against `buy_and_hold` in the same table. Also inspect the
per-fold table: a headline Sharpe built on two or three trades is noise, not a result.
`consistency` below `0.5` means the folds disagree, which usually means the edge is
absent rather than unstable.

**4. Leakage and robustness.** The suite pins the invariants that make the numbers
trustworthy — next-bar fills, no retroactive effect from appending future bars, fold
disjointness, and no leverage in the sizing engine. Re-run `py -m pytest -q` after any
change to `backtest.py` before reading new numbers.

**Interpreting the bundled model.** The model in the local registry scores 0.5003
directional accuracy and +0.0031 correlation, and 99% of its predictions are negative.
It is a working baseline, not a usable forecast.

**Interpreting a sweep.** A full `1h`/`4h`/`1d` x `lstm`/`transformer`/`mlp` run over
365 days of BTC cleared **zero** of nine cells. Directional accuracy sat between 0.457
and 0.543, correlation between −0.083 and +0.043, and no cell produced more than three
walk-forward trades. Read that as evidence the current features and labels carry no
edge, not as a tuning problem: the next step is a different hypothesis (label, feature
set, or data), not more epochs. The daily cells cannot be trained at all on one year of
history and are reported as `insufficient_data`.

---

## Methodology notes

These are the correctness properties the test suite pins down. They are the parts most
likely to be quietly wrong in a quant pipeline.

**No lookahead.** A signal produced at bar `i` is filled at `open[i+1]`, never at a price
that was only knowable afterwards. `trades` carry a `signal_index` so the lag is auditable.
Appending extra future bars cannot change any past result.

**Causal features.** Every rolling window, resample and backfill uses trailing data only.
Sentiment resamples use `closed='left'`, so a bar sees only articles published strictly
before its close. Fear & Greed readings published at midnight become visible at that bar,
not the previous one.

**Walk-forward validation.** Each fold refits on its own training window and forecasts the
unseen test slice. Train-length signals are rejected rather than padded. Benchmarks are
computed from the fold's own test prices, and per-fold returns are chained so CAGR reflects
the actual out-of-sample period.

**Fractional execution.** The RL environments default to `allow_fractional=True` with
venue-level `quantity_precision`, so a 10k account can trade BTC instead of only
alt-coins. Fees are charged on the executed notional, and a ruin guard terminates a
wiped-out episode.

**Capital-normalized sizing.** `BacktestEngine.run_signal_strategy` treats
`position_size` as a fraction of *equity committed as notional*, clamped to `[0, 1]`,
matching the RL environments' `min(1.0, position_size)`. Sizing is marked at the fill
price, not the signal-bar close, so a `position_size=1.0` position earns the
instrument's own return bar for bar instead of drifting to ~0.99 exposure. Sizing in
whole asset units instead would put 1 BTC (~$76k) behind a $10k account, drive equity
negative on adverse signals, and report impossible drawdowns below -100%. Two
consequences to keep in mind when reading output: a max drawdown can no longer exceed
-100%, and `max_drawdown`/`cagr` from before that change are not comparable.

**Interval alignment.** `sliding_window_view` appends its window axis, so inference windows
are transposed to `(batch, seq, features)` — the same layout training feeds the model.
Predictions align to `frame.iloc[seq_len:]`, and the extra trailing window is dropped.

**Timezones.** SQLite datetime columns use a `UTCDateTime` decorator: values are stored as
naive UTC and always read back timezone-aware. Mixing naive and aware comparisons is a
common source of intermittent, data-dependent failures.

**Uncertainty.** Forecaster heads are Gaussian, so every prediction carries `lower_68` /
`upper_68` bands and a calibration-aware confidence. Coverage is reported, not assumed.

---

## Testing

```bash
py -m pytest -q
py -m ruff check src tests
```

The suite is offline and hermetic: provider parsers are tested against captured fixtures,
and every test writes to a temporary data directory. Nothing touches the real `data/`.

---

## Project layout

```
cqip/
├── src/cqip/            Application package
├── tests/
│   ├── conftest.py      Isolated temp storage + deterministic fixtures
│   ├── test_data_and_features.py
│   ├── test_models_and_backtest.py
│   ├── test_providers.py
│   └── test_api.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Status and limitations

- Ingestion, forecasting, RL, backtesting, the registry and the API are functional end to end.
- Backtested models on the bundled sample data have **not** been profitable. The
  registered model has near-chance directional accuracy and a collapsed prediction
  distribution, so treat its output as a plumbing baseline and never as performance
  evidence. See [Evaluating model quality](#evaluating-model-quality).
- The job manager is an in-process thread pool, so long jobs die with the server. Move it
  to a real queue before running multiple workers.
- Price streaming polls stored Parquet every 5s rather than consuming exchange WebSockets,
  which avoids API-key requirements at the cost of latency.
- The frontend is not part of this phase.
