# Ship derived forecasts, not the input panel

The raw inputs are licensed (Refinitiv, SETSMART) and cannot be redistributed,
which would normally leave a reader unable to verify a single published number.
We commit `data/processed/oos_forecasts.parquet` (~400 KB) because it contains
only derived statistics — forward realized betas, the three forecasts, log
market cap, and the stock-month key — from which no licensed price series can be
reconstructed, and because it is enough to reproduce the entire evidence base:
the VW-MSE table, the Diebold–Mariano horse race, and the size and regime cuts.

## Amendment: the importance table

`data/processed/predictor_importance.csv` (10 rows: predictor, VI, t-stat,
p-value) is committed on the same grounds. It is derived statistics like the
forecast table, a handful of loss differentials from which no licensed series
can be reconstructed. Committing it makes every Case Study number reproducible
from a bare clone, so the showcase export never needs the trained models.

## Considered options

`lstm_input_panel.parquet` (3 MB) was rejected despite making training
reproducible: ILLIQ, TO, IVOL and DIVPAY are firm characteristics computed
directly from licensed price and volume data, so redistributing the panel is
much closer to redistributing the source. A synthetic-data generator was also
rejected — it would prove the code runs, but nothing about the result.

## Consequences

Evaluation is reproducible; training is not, and the README says so plainly
rather than implying otherwise. `betaforecast.importance.subsamples` is split so
its accuracy half runs without the models, and the steps that do need them fail
with an explicit message. Publication is irreversible: if the licence position
on derived statistics ever changes, the file is already public.
