# Forecasting Stock Betas with an LSTM (SET100)

Can a neural network predict a stock's **future market beta** better than the
rolling-regression estimates the industry has used for decades? This project
builds a walk-forward **LSTM** that forecasts the one-year-ahead *realized beta*
of Thai SET100 stocks out-of-sample, and benchmarks it head-to-head against the
two classic rolling-OLS estimators using a proper Diebold–Mariano test.

> **Beta** measures how much a stock moves with the market — the key input to the
> cost of equity (CAPM), portfolio hedging, and risk budgeting. The catch: the
> beta you *want* is next year's, but all you can measure is the past. This repo
> treats "next year's beta" as a supervised-learning target and forecasts it.

## Headline result

Out-of-sample, 2015–2024 (120 months, 10,145 stock-month forecasts), scored on
**value-weighted MSE** (market-cap weighted, so large stocks count more — the way
a portfolio actually holds them):

| Model | What it is | VW-MSE | vs LSTM |
|---|---|---:|---:|
| **LSTM** | 8-unit network over a 12-month lookback of 10 predictors | **0.0995** | — |
| OLS&nbsp;1Y&nbsp;daily | 1 year of daily returns, rolling OLS beta | 0.0974 | −0.0021 (tie) |
| OLS&nbsp;5Y&nbsp;monthly | 5 years of monthly returns, rolling OLS beta | 0.1968 | +0.0973 |

- The LSTM **matches** the strong short-window daily benchmark (Diebold–Mariano
  *t* = 0.20, not significant) and **decisively beats** the long-window monthly
  benchmark (**DM *t* = −6.55, p < 0.001**) — the gap holds pre- and post-COVID.
- A zero-out importance test shows the forecast leans on **recent realized betas**
  (`OLS_3M_D` *t* = 4.93, `OLS_1Y_D` *t* = 3.58); firm characteristics add little.

Takeaway: an LSTM learns to reproduce the accuracy of the best simple estimator
and to avoid the large errors of the slow-moving one — without being told which
benchmark to trust.

## Approach

```
 raw SET100 data ──► panel of (stock, month): target + 10 predictors
                          │
                          ▼
        12-month lookback sequences  (strict, gap-free, no look-ahead)
                          │
                          ▼
     walk-forward LSTM ensemble  (10 rolling refits × 10 random seeds)
                          │
                          ▼
   out-of-sample forecasts ──► value-weighted MSE + Diebold–Mariano tests
```

Design choices that make the horse race fair:

- **Realized-beta target** — beta actually realized over the *next* 12 months, so
  the model is graded on the future, not the past.
- **Value-weighted loss and metric** — training and evaluation both weight each
  stock by market cap, matching how the forecast would be used.
- **Walk-forward with a gap year** — the model is refit each year on only past
  data; a one-year buffer between the validation and test windows removes the
  overlap that the 12-month horizon would otherwise create (no look-ahead).
- **Seed ensemble** — 10 random initialisations averaged, to cancel the noise of
  any single unlucky fit.
- **Honest significance test** — Diebold–Mariano with a Newey–West HAC standard
  error at lag 11, because 12-month-ahead errors overlap by construction.

## Read the notebooks in order

The `notebooks/` walkthrough is the best way in — each one renders its own plots
and tables:

1. **[`01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb)** — from raw
   prices/dividends to the clean stock-month panel of target + 10 predictors.
2. **[`02_lstm_explained.ipynb`](notebooks/02_lstm_explained.ipynb)** — a code
   walkthrough of the model: lookback sequences → leak-free standardisation →
   the value-weighted walk-forward LSTM, with a runnable mini-demo.
3. **[`03_oos_forecast_plots.ipynb`](notebooks/03_oos_forecast_plots.ipynb)** — the
   out-of-sample results: forecast-vs-realized beta and value-weighted MSE over time.

## Repository layout

```
betaforecast/               the pipeline, grouped by stage (importable package)
├── data/                   load raw inputs → build the target + 10 predictors
│   ├── sources.py          raw loaders (prices, factors, membership, ...)
│   ├── dividends.py        cash-dividend yields → total returns
│   ├── returns.py          daily / monthly total & excess returns
│   ├── predictors.py       ME, DIVPAY, TO, MOM, IVOL, ILLIQ, DFY
│   ├── panel.py            realized-beta target + rolling-OLS betas
│   └── build.py            end-to-end build + descriptive tables
├── features/               model-ready inputs
│   ├── sequences.py        strict 12-month lookback windows + walk-forward splits
│   └── standardize.py      per-month rank-standardisation + per-fold DFY scaling
├── model/
│   ├── lstm.py             the LSTM, value-weighted training, walk-forward driver
│   ├── tune.py             hyperparameter grid search (offline experiment)
│   └── analyze_tuning.py   summarise the tuning run
├── benchmarks.py           align the two rolling-OLS benchmarks to the OOS set
├── evaluation/
│   ├── evaluate.py         value-weighted MSE
│   └── dm_test.py          Diebold–Mariano tests (full sample + pre/post-COVID)
├── importance/             zero-out predictor importance, size/regime cuts
├── figures/                figure-generation scripts
└── paths.py                one place for all filesystem paths
notebooks/                  the three-part walkthrough above
```

## Setup

Uses [uv](https://docs.astral.sh/uv/) with a pinned, reproducible environment
(TensorFlow / Keras 3, pandas, numpy, scipy, matplotlib):

```bash
uv sync                 # install everything, incl. the notebook tooling
uv run jupyter lab      # open the notebooks
```

Run any pipeline stage as a module from the repo root, e.g.:

```bash
uv run python -m betaforecast.model.lstm        # train the walk-forward ensemble
uv run python -m betaforecast.evaluation.evaluate   # value-weighted MSE table
uv run python -m betaforecast.evaluation.dm_test    # Diebold–Mariano horse race
```

Typical order: `data.build` → `model.lstm` → `benchmarks` → `evaluation.evaluate`
→ `evaluation.dm_test` → `importance.zero_out` → the `figures/` scripts.

## Data availability

The raw data (Refinitiv prices, SETSMART dividends, SET100 membership), the
trained model checkpoints, and the paper are **not** distributed (licensing and
size). The code expects raw inputs under `data/raw/` and writes processed panels,
forecasts, and tables under `data/processed/`; reproducing the numbers requires
those inputs and a full pipeline run.

## Method notes

- **Rank standardisation** — each month, the nine firm/beta predictors are ranked
  cross-sectionally and mapped to [−1, 1] (following Gu, Kelly & Xiu 2020), so the
  model sees comparable, outlier-robust inputs with no look-ahead.
- **Rolling vs expanding window** — the training window rolls (5 years) rather than
  expands, following Drobetz et al. (2025); an expanding window tested slightly
  worse.
- **Benchmarks are persistence forecasts** — the rolling-OLS betas are used as-is
  as next-year forecasts, so there is no extra estimation to tune.
