"""Regression pin for the tercile x regime accuracy report (value-weighted).

The numbers were captured from `tercile_regime_dm` before it was generalised
with a `weighted` flag; the refactor must leave them, the column order and the
row order untouched.
"""
import pytest

from betaforecast.importance.subsamples import tercile_regime_dm

COLUMNS = ["DM_OLS_1Y_D", "p_OLS_1Y_D", "MSE_LSTM", "MSE_OLS_1Y_D",
           "DM_OLS_5Y_M", "p_OLS_5Y_M", "MSE_OLS_5Y_M"]

PINNED = {
    ("Small", "pre"): [-5.573401, 0.0, 0.088549, 0.150253, -5.991348, 0.0, 0.380197],
    ("Small", "post"): [-0.540031, 0.589175, 0.164706, 0.183742, -7.284903, 0.0, 0.338851],
    ("Mid", "pre"): [-4.60949, 4e-06, 0.101483, 0.142422, -6.111279, 0.0, 0.246006],
    ("Mid", "post"): [-1.04079, 0.297973, 0.1119, 0.129176, -6.169652, 0.0, 0.204437],
    ("Large", "pre"): [0.744063, 0.456838, 0.091778, 0.081843, -4.364805, 1.3e-05, 0.136621],
    ("Large", "post"): [0.602001, 0.547174, 0.100288, 0.088757, -4.5368, 6e-06, 0.226269],
}


def test_tercile_regime_report_is_unchanged(oos):
    got = tercile_regime_dm(oos)
    assert list(got.columns) == COLUMNS
    assert list(got.index) == list(PINNED)
    for key, expected in PINNED.items():
        assert got.loc[key].tolist() == pytest.approx(expected, abs=2e-6)
