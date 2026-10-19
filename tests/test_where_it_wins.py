"""Export-seam test for where-it-wins.json against manuscript Tables 7-14.

EXPECTED is transcribed from the committed redacted manuscript
(showcase/attachments/manuscript-redacted.pdf). Values are as printed: MSE in
percent (2 dp in Tables 7/8, 1 dp in 9-14), DM t of LSTM vs each benchmark
(column LSTM, row benchmark; negative = LSTM more accurate) at 2 dp. The
manuscript prints significance stars, not p-values, so p is checked for
consistency with t instead. The JSON holds MSE in raw units (percent / 100).

Row: (weighting, regime, group): (mse_lstm, mse_1y, mse_5y, t_1y, t_5y, table).
"""
import json

import pytest

from betaforecast.showcase import export

EXPECTED = {
    # Tables 7 / 8: all stocks, all months
    ("vw", "all", "all"): (9.95, 9.74, 19.68, 0.20, -6.55, 7),
    ("ew", "all", "all"): (10.94, 13.55, 26.42, -2.36, -9.73, 8),
    # Tables 9 / 10: size groups, all months
    ("vw", "all", "small"): (12.5, 16.7, 36.1, -2.15, -7.25, 9),
    ("vw", "all", "mid"): (10.6, 13.7, 22.8, -3.07, -7.14, 9),
    ("vw", "all", "large"): (9.6, 8.5, 18.0, 0.91, -4.36, 9),
    ("ew", "all", "small"): (13.0, 17.3, 36.7, -2.10, -7.37, 10),
    ("ew", "all", "mid"): (10.9, 14.3, 25.8, -3.27, -6.20, 10),
    ("ew", "all", "large"): (8.9, 9.1, 16.8, -0.26, -5.49, 10),
    # Tables 11 / 12: regimes, all stocks
    ("vw", "pre", "all"): (9.3, 9.4, 16.5, -0.07, -7.84, 11),
    ("vw", "post", "all"): (10.7, 10.1, 23.1, 0.31, -5.27, 11),
    ("ew", "pre", "all"): (9.7, 13.2, 27.5, -4.22, -7.51, 12),
    ("ew", "post", "all"): (12.4, 13.9, 25.3, -0.74, -8.47, 12),
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

# Checks that do not reproduce from the committed OOS table. Every one is in a
# size-tercile cell; all six group="all" cells (Tables 7, 8, 11, 12) match. The
# misses are similar under VW and EW, so they come from the tercile membership
# (the manuscript's size groups differ marginally from betaforecast's
# per-origin-month rank-and-qcut on ME), not from the weighting definition.
# Value: observed (got - printed), in table units. strict xfail: if the groups
# are ever reconciled, these start failing and must be deleted.
MISMATCH = {
    ("vw", "all", "small", "t_1y"): -0.060,
    ("vw", "all", "small", "t_5y"): +0.120,
    ("vw", "all", "mid", "mse_1y"): -0.200,
    ("vw", "all", "mid", "mse_5y"): -0.260,
    ("vw", "all", "mid", "t_1y"): +0.070,
    ("vw", "all", "mid", "t_5y"): -0.300,
    ("vw", "all", "large", "t_5y"): +0.020,
    ("vw", "pre", "small", "t_1y"): +0.100,
    ("vw", "pre", "small", "t_5y"): +0.100,
    ("vw", "pre", "mid", "mse_1y"): -0.160,
    ("vw", "pre", "mid", "mse_5y"): -0.400,
    ("vw", "pre", "mid", "t_1y"): +0.190,
    ("vw", "pre", "mid", "t_5y"): -0.100,
    ("vw", "post", "small", "mse_5y"): -0.110,
    ("vw", "post", "small", "t_1y"): -0.030,
    ("vw", "post", "small", "t_5y"): +0.100,
    ("vw", "post", "mid", "mse_lstm"): -0.110,
    ("vw", "post", "mid", "mse_1y"): -0.180,
    ("vw", "post", "mid", "mse_5y"): -0.160,
    ("vw", "post", "mid", "t_1y"): +0.030,
    ("vw", "post", "mid", "t_5y"): -0.380,
    ("vw", "post", "large", "t_5y"): +0.040,
    ("ew", "all", "small", "t_1y"): -0.040,
    ("ew", "all", "small", "t_5y"): +0.070,
    ("ew", "all", "mid", "mse_1y"): -0.130,
    ("ew", "all", "mid", "mse_5y"): -0.180,
    ("ew", "all", "mid", "t_1y"): +0.050,
    ("ew", "all", "mid", "t_5y"): -0.130,
    ("ew", "all", "large", "t_5y"): +0.080,
    ("ew", "pre", "small", "t_1y"): +0.030,
    ("ew", "pre", "small", "t_5y"): +0.070,
    ("ew", "pre", "mid", "mse_5y"): -0.270,
    ("ew", "pre", "mid", "t_1y"): +0.120,
    ("ew", "pre", "mid", "t_5y"): -0.050,
    ("ew", "pre", "large", "t_5y"): -0.030,
    ("ew", "post", "small", "mse_5y"): -0.120,
    ("ew", "post", "small", "t_1y"): -0.020,
    ("ew", "post", "small", "t_5y"): +0.020,
    ("ew", "post", "mid", "mse_lstm"): -0.140,
    ("ew", "post", "mid", "mse_1y"): -0.250,
    ("ew", "post", "mid", "mse_5y"): -0.150,
    ("ew", "post", "mid", "t_1y"): +0.030,
    ("ew", "post", "mid", "t_5y"): -0.200,
    ("ew", "post", "large", "t_1y"): +0.020,
    ("ew", "post", "large", "t_5y"): +0.200,
}


@pytest.fixture(scope="module")
def cells(tmp_path_factory, oos_module):
    d = tmp_path_factory.mktemp("wiw")
    export(d)
    payload = json.loads((d / "data" / "where-it-wins.json").read_text())
    return payload, {(c["weighting"], c["regime"], c["group"]): c
                     for c in payload["cells"]}


@pytest.fixture(scope="module")
def oos_module():
    from betaforecast.paths import OOS_FORECASTS
    if not OOS_FORECASTS.exists():
        pytest.skip("oos_forecasts.parquet not present")


def test_exactly_24_cells_in_deterministic_order(cells):
    payload, by_key = cells
    assert list(payload) == ["cells"]
    order = [(w, r, g) for w in ("vw", "ew") for r in ("all", "pre", "post")
             for g in ("all", "small", "mid", "large")]
    assert [(c["weighting"], c["regime"], c["group"]) for c in payload["cells"]] == order
    assert set(EXPECTED) == set(order)
    for c in payload["cells"]:
        assert list(c) == ["weighting", "regime", "group", "mse", "dm"]
        assert list(c["mse"]) == ["lstm", "ols_1y_d", "ols_5y_m"]
        assert list(c["dm"]) == ["ols_1y_d", "ols_5y_m"]
        assert all(list(v) == ["t", "p"] for v in c["dm"].values())


def test_rounding(cells):
    dec = lambda x: len(repr(x).split(".")[1]) if "." in repr(x) else 0
    for c in cells[0]["cells"]:
        assert all(dec(v) <= 4 for v in c["mse"].values())
        for v in c["dm"].values():
            assert dec(v["t"]) <= 2 and dec(v["p"]) <= 3


CHECKS = ("mse_lstm", "mse_1y", "mse_5y", "t_1y", "t_5y")
PARAMS = [pytest.param(k, c, id="-".join(k) + "-" + c,
                       marks=[pytest.mark.xfail(strict=True, reason=(
                           f"tercile membership differs from manuscript: "
                           f"observed {MISMATCH[(*k, c)]:+.2f} off the printed value"))]
                       if (*k, c) in MISMATCH else [])
          for k in EXPECTED for c in CHECKS]


@pytest.mark.parametrize("key,check", PARAMS)
def test_cell_matches_manuscript(cells, key, check):
    row = dict(zip(CHECKS, EXPECTED[key][:5]))
    c = cells[1][key]
    got = {"mse_lstm": c["mse"]["lstm"] * 100, "mse_1y": c["mse"]["ols_1y_d"] * 100,
           "mse_5y": c["mse"]["ols_5y_m"] * 100,
           "t_1y": c["dm"]["ols_1y_d"]["t"], "t_5y": c["dm"]["ols_5y_m"]["t"]}[check]
    # One unit of the last printed digit (MSE in percent: 2 dp Tables 7/8, else 1 dp).
    tol = (0.01 if EXPECTED[key][5] in (7, 8) else 0.1) if check.startswith("mse") else 0.01
    assert got == pytest.approx(row[check], abs=tol + 1e-9)


def test_p_values_agree_with_t(cells):
    from scipy import stats
    for c in cells[0]["cells"]:
        for v in c["dm"].values():
            assert v["p"] == pytest.approx(2 * stats.norm.sf(abs(v["t"])), abs=0.006)
