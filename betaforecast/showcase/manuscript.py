"""The manuscript's printed size-group cells (source of truth for the Case Study).

``MANUSCRIPT_SIZE_CELLS`` transcribes the size-group cells of the redacted
manuscript (``showcase/attachments/manuscript-redacted.pdf``):

* Table 9  - size groups, all months, value-weighted
* Table 10 - size groups, all months, equal-weighted
* Table 13 - size groups x pre/post-COVID, value-weighted
* Table 14 - size groups x pre/post-COVID, equal-weighted

Keys are ``(weighting, regime, group)`` with weighting in {"vw", "ew"}, regime
in {"all", "pre", "post"} and group in {"small", "mid", "large"}. Values are
``(mse_lstm, mse_ols_1y_d, mse_ols_5y_m, t_ols_1y_d, t_ols_5y_m, table)``:
MSE in percent at the printed 1 dp, and the Diebold-Mariano t of the LSTM
(column) against each benchmark (row) at the printed 2 dp; negative means the
LSTM is more accurate. The manuscript prints significance stars, not p-values.

Why these are transcribed rather than computed: the manuscript's size-group
assignment is not recoverable from the committed data, so recomputing the cells
differs slightly from the printed tables (see
docs/adr/0002-manuscript-size-tables-are-source-of-truth.md). The all-stock
cells (group "all") are not in this table; they reproduce exactly and are
always computed.
"""
from scipy import stats

# Raw-unit precision of the manuscript's MSE (percent at 1 dp -> 3 dp raw).
DP_MSE = 3
DP_T, DP_P = 2, 3

MANUSCRIPT_SIZE_CELLS = {
    # Table 9 / 10: size groups, all months
    ("vw", "all", "small"): (12.5, 16.7, 36.1, -2.15, -7.25, 9),
    ("vw", "all", "mid"): (10.6, 13.7, 22.8, -3.07, -7.14, 9),
    ("vw", "all", "large"): (9.6, 8.5, 18.0, 0.91, -4.36, 9),
    ("ew", "all", "small"): (13.0, 17.3, 36.7, -2.10, -7.37, 10),
    ("ew", "all", "mid"): (10.9, 14.3, 25.8, -3.27, -6.20, 10),
    ("ew", "all", "large"): (8.9, 9.1, 16.8, -0.26, -5.49, 10),
    # Table 13: size x regime, value-weighted
    ("vw", "pre", "small"): (8.9, 15.1, 38.1, -5.67, -6.09, 13),
    ("vw", "post", "small"): (16.4, 18.3, 34.0, -0.51, -7.38, 13),
    ("vw", "pre", "mid"): (10.2, 14.4, 25.0, -4.80, -6.01, 13),
    ("vw", "post", "mid"): (11.3, 13.1, 20.6, -1.07, -5.79, 13),
    ("vw", "pre", "large"): (9.2, 8.2, 13.7, 0.75, -4.37, 13),
    ("vw", "post", "large"): (10.0, 8.9, 22.6, 0.60, -4.58, 13),
    # Table 14: size x regime, equal-weighted
    ("ew", "pre", "small"): (9.8, 16.2, 38.7, -4.79, -6.49, 14),
    ("ew", "post", "small"): (16.5, 18.2, 34.4, -0.47, -6.34, 14),
    ("ew", "pre", "mid"): (10.1, 14.7, 29.1, -6.97, -5.45, 14),
    ("ew", "post", "mid"): (12.0, 14.1, 22.7, -1.11, -5.44, 14),
    ("ew", "pre", "large"): (9.2, 8.8, 14.8, 0.29, -3.75, 14),
    ("ew", "post", "large"): (8.7, 9.5, 18.9, -0.64, -4.94, 14),
}


def p_from_t(t):
    """Two-sided normal p-value, as in ``evaluation.dm_test.dm_stat``."""
    return float(2 * stats.norm.sf(abs(t)))


def stars(p):
    """Significance level as the manuscript's stars: 3 (1%), 2 (5%), 1 (10%), 0."""
    return 3 if p < 0.01 else 2 if p < 0.05 else 1 if p < 0.10 else 0


def manuscript_cell(key):
    """JSON-ready ``mse`` and ``dm`` dicts for a transcribed cell, in raw units.

    MSE is percent / 100 rounded to the printed precision (3 dp); t is as
    printed; p is computed from the printed t (stars are all the manuscript
    prints) and rounded to 3 dp.
    """
    lstm, m1, m5, t1, t5, _ = MANUSCRIPT_SIZE_CELLS[key]
    raw = lambda pct: round(pct / 100, DP_MSE) + 0.0
    dm = lambda t: {"t": round(t, DP_T) + 0.0, "p": round(p_from_t(t), DP_P) + 0.0}
    return ({"lstm": raw(lstm), "ols_1y_d": raw(m1), "ols_5y_m": raw(m5)},
            {"ols_1y_d": dm(t1), "ols_5y_m": dm(t5)})
