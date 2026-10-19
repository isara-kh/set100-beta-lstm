"""Export-seam test for where-it-wins.json against manuscript Tables 7-14.

The six group="all" cells (Tables 7, 8, 11, 12) are transcribed in EXPECTED from
the committed redacted manuscript (showcase/attachments/manuscript-redacted.pdf)
as an independent check of the computed numbers: MSE in percent (2 dp in
Tables 7/8, 1 dp in 11/12), DM t of LSTM vs each benchmark at 2 dp (negative =
LSTM more accurate). The 18 size-group cells (Tables 9, 10, 13, 14) are the
manuscript's own and live in betaforecast.showcase.manuscript (ADR-0002); the
export must equal them exactly. The JSON holds MSE in raw units (percent / 100).

EXPECTED row: (mse_lstm, mse_1y, mse_5y, t_1y, t_5y, table).
"""
import json

import pytest

from betaforecast.showcase import export
from betaforecast.showcase.manuscript import (
    MANUSCRIPT_SIZE_CELLS, p_from_t, stars)

EXPECTED = {
    # Tables 7 / 8: all stocks, all months
    ("vw", "all", "all"): (9.95, 9.74, 19.68, 0.20, -6.55, 7),
    ("ew", "all", "all"): (10.94, 13.55, 26.42, -2.36, -9.73, 8),
    # Tables 11 / 12: regimes, all stocks
    ("vw", "pre", "all"): (9.3, 9.4, 16.5, -0.07, -7.84, 11),
    ("vw", "post", "all"): (10.7, 10.1, 23.1, 0.31, -5.27, 11),
    ("ew", "pre", "all"): (9.7, 13.2, 27.5, -4.22, -7.51, 12),
    ("ew", "post", "all"): (12.4, 13.9, 25.3, -0.74, -8.47, 12),
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
    assert set(EXPECTED) | set(MANUSCRIPT_SIZE_CELLS) == set(order)
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


def _json_row(c):
    return (c["mse"]["lstm"], c["mse"]["ols_1y_d"], c["mse"]["ols_5y_m"],
            c["dm"]["ols_1y_d"]["t"], c["dm"]["ols_5y_m"]["t"])


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_all_stock_cells_match_manuscript_within_rounding(cells, key):
    """Independent check of the computed numbers: the six group="all" cells."""
    lstm, m1, m5, t1, t5, table = EXPECTED[key]
    tol = 0.01 if table in (7, 8) else 0.1          # last printed digit, percent
    got = _json_row(cells[1][key])
    for g, want, tl in zip(got[:3], (lstm, m1, m5), (tol,) * 3):
        assert g * 100 == pytest.approx(want, abs=tl + 1e-9)
    for g, want in zip(got[3:], (t1, t5)):
        assert g == pytest.approx(want, abs=0.01 + 1e-9)


@pytest.mark.parametrize("key", sorted(MANUSCRIPT_SIZE_CELLS))
def test_size_cells_equal_manuscript_exactly(cells, key):
    lstm, m1, m5, t1, t5, _ = MANUSCRIPT_SIZE_CELLS[key]
    c = cells[1][key]
    assert c["mse"] == {"lstm": round(lstm / 100, 3), "ols_1y_d": round(m1 / 100, 3),
                        "ols_5y_m": round(m5 / 100, 3)}
    assert (c["dm"]["ols_1y_d"]["t"], c["dm"]["ols_5y_m"]["t"]) == (t1, t5)


def test_there_are_18_size_cells_and_6_all_stock_cells():
    assert len(MANUSCRIPT_SIZE_CELLS) == 18 and len(EXPECTED) == 6
    assert not set(EXPECTED) & set(MANUSCRIPT_SIZE_CELLS)


def test_p_values_agree_with_t(cells):
    from scipy import stats
    for c in cells[0]["cells"]:
        for v in c["dm"].values():
            assert v["p"] == pytest.approx(2 * stats.norm.sf(abs(v["t"])), abs=0.006)


def test_size_cell_p_is_exactly_normal_from_t(cells):
    from scipy import stats
    for key in MANUSCRIPT_SIZE_CELLS:
        for v in cells[1][key]["dm"].values():
            assert v["p"] == round(2 * stats.norm.sf(abs(v["t"])), 3)


def test_computed_size_results_agree_with_manuscript_qualitatively(oos_module):
    """Guard against code drift: the repo's own tercile recomputation differs
    slightly from the printed tables (ADR-0002) but must agree on sign,
    significance level (10/5/1%) and the lowest-MSE model in every cell."""
    import pandas as pd
    from betaforecast.importance import subsamples
    from betaforecast.paths import OOS_FORECASTS
    oos = pd.read_parquet(OOS_FORECASTS)
    for w in ("vw", "ew"):
        computed = subsamples.group_regime_cells(oos, weighted=(w == "vw"))
        for (g, r), c in computed.items():
            if g == "all":
                continue
            lstm, m1, m5, t1, t5, _ = MANUSCRIPT_SIZE_CELLS[(w, r, g)]
            got_mse = (c["MSE_LSTM"], c["OLS_1Y_D"]["MSE"], c["OLS_5Y_M"]["MSE"])
            want_mse = (lstm, m1, m5)
            assert got_mse.index(min(got_mse)) == want_mse.index(min(want_mse)), (w, r, g)
            for name, t in (("OLS_1Y_D", t1), ("OLS_5Y_M", t5)):
                assert (c[name]["DM"] < 0) == (t < 0), (w, r, g, name)
                assert stars(c[name]["p"]) == stars(p_from_t(t)), (w, r, g, name)
