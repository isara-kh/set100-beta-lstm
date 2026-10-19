# The manuscript's size-group tables are the source of truth for the Case Study

The 18 size-group cells of the Case Study's where-it-wins embed (small, mid and
large, across all months, pre- and post-COVID, value- and equal-weighted) come
from the manuscript's printed Tables 9, 10, 13 and 14, transcribed in
`betaforecast/showcase/manuscript.py`. They are not recomputed at export. The
six all-stock cells (Tables 7, 8, 11, 12) reproduce exactly and stay computed.

## Why

Recomputing the size groups from the committed `oos_forecasts.parquet` with the
repo's per-origin-month terciles on market cap differs slightly from the printed
tables: MSE by up to 0.40 percentage points and the Diebold-Mariano t by up to
0.38. The manuscript's original size-group assignment is not recoverable from
the committed data, so the gap cannot be closed. The manuscript is the
published result, so the owner decided it wins.

## What stays computed

The README tables and the `by_size` / `subsamples` reports keep documenting what
the code produces from the committed data. They are not changed by this
decision.

## Consequences

The direction, significance level and lowest-MSE model of every cell agree
between the manuscript and the recomputation, and a test keeps it so.
Transcribed MSEs are raw units at the printed precision (percent / 100, 3 dp).
The manuscript prints only stars, so p is computed from the printed t with a
two-sided normal, as `evaluation/dm_test.py` does, at 3 dp. The export JSON has
no provenance field (the site schema is strict): a cell is transcribed exactly
when its group is small, mid or large.
