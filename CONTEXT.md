# Beta Forecasting

Out-of-sample forecasting of Thai SET100 stock betas: a walk-forward LSTM is
graded against rolling-OLS beta estimators on a value-weighted loss, with
Diebold–Mariano tests establishing whether any accuracy difference is real.

## Language

### The quantity being forecast

**Forward realized beta**:
The beta a stock actually realizes over the twelve months *after* a forecast
origin. This is the prediction target, never an input.
_Avoid_: realized beta, future beta, next-year beta

**Rolling beta**:
A backward-looking OLS beta estimated over a window ending at the origin, named
by window and frequency (`OLS_3M_D`, `OLS_1Y_D`, `OLS_5Y_M`). Serves two roles:
predictor, and — used unchanged as a forecast — benchmark.
_Avoid_: historical beta, estimated beta, realized beta

**Market proxy**:
The SET100 index return used as the market leg of the beta regression.
_Avoid_: benchmark, market benchmark, the index

### The horse race

**Benchmark**:
A competing forecast the LSTM is scored against — here `OLS_1Y_D` and
`OLS_5Y_M`. Follows Diebold–Mariano usage. Never refers to the market proxy.
_Avoid_: baseline, comparison model, reference forecast

**Persistence forecast**:
Using a rolling beta unchanged as the forecast of forward realized beta. Both
benchmarks are persistence forecasts, so neither has parameters to tune.

**Value-weighted MSE**:
The primary metric. Within each origin month, squared errors are weighted by
market-cap share; the reported figure is the equal-weighted average across
origin months. Weights come from raw ME, never the rank-standardised predictor.
_Avoid_: VW-MSE as a synonym for pooled weighted MSE

### Sample construction

**Forecast origin**:
The month *t* from which a forecast of forward realized beta is made. All
subsample cuts — size, regime — are made on the origin, never on the
realization month.
_Avoid_: prediction date, base month

**Member-month**:
A stock-month in which the stock belongs to the SET100. A forecast origin must
be a member-month; interior lookback months need not be.

**Lookback window**:
The twelve consecutive, gap-free months ending at the origin that form one
input sequence.
_Avoid_: sequence length, history window

**Walk-forward iteration**:
One refit step of the rolling design. Iteration *k* trains on the five origin
years 2008+k…2012+k, validates on 2013+k, discards 2014+k, and tests on 2015+k.

**Gap year**:
The discarded year between validation and test. Its length equals the twelve
month forecast horizon, so validation labels' realization windows cannot
overlap the test window — early stopping therefore carries no look-ahead.
_Avoid_: buffer, embargo

### Subsample cuts

**Size tercile**:
Small / Mid / Large, assigned by ranking stocks on ME *within each origin
month*, so the grouping is relative and re-formed every month.
_Avoid_: size bucket, cap group

**Regime**:
The pre/post-COVID split on the forecast origin — pre ≤ 2019-12, post ≥
2020-03. The two origins inside the Thai crisis window are dropped, so the
split contrasts information regimes rather than calm against turbulent.

**Zero-out importance**:
Predictor importance measured by muting one predictor in the trained ensemble
and rescoring; the reported value is the resulting rise in value-weighted MSE.
_Avoid_: feature importance, permutation importance
