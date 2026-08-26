"""Weighting and test-statistic invariants, plus a regression test that pins
the published headline numbers to the shipped forecast table.
"""
import numpy as np
import pandas as pd
import pytest

from betaforecast.evaluation import dm_test, evaluate


def test_dm_lag_equals_horizon_minus_one():
    """Overlapping 12-month errors are MA(11); the lag is structural, not tuned."""
    assert dm_test.LAG == dm_test.H - 1 == 11


def test_dm_sign_convention_favours_the_lstm_when_negative():
    """d_t = loss(LSTM) - loss(benchmark), so a negative mean means LSTM wins."""
    rng = np.random.default_rng(0)
    d = pd.Series(rng.normal(loc=-1.0, scale=0.1, size=120))
    dbar, dm, p, T = dm_test.dm_stat(d)
    assert dbar < 0 and dm < 0 and T == 120


def test_dm_statistic_is_zero_for_identical_forecasts():
    d = pd.Series(np.zeros(120))
    with np.errstate(invalid="ignore"):        # 0/0 is the point of the case
        dbar, dm, p, T = dm_test.dm_stat(d)
    assert dbar == 0.0
    assert np.isnan(dm) or dm == 0.0


def test_hac_variance_exceeds_iid_variance_under_positive_autocorrelation():
    """The whole point of the HAC correction: overlapping errors inflate the SE."""
    rng = np.random.default_rng(1)
    e = rng.normal(size=400)
    d = pd.Series(np.convolve(e, np.ones(12), mode="valid"))   # MA(11) by construction
    _, dm_hac, _, T = dm_test.dm_stat(d)
    se_iid = d.std(ddof=0) / np.sqrt(T)
    dm_iid = d.mean() / se_iid
    assert abs(dm_hac) < abs(dm_iid)


def test_value_weights_sum_to_one_within_each_origin_month(oos):
    df = evaluate.load()
    sums = df.groupby("ym")["w"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_weights_come_from_raw_market_cap_not_the_standardised_predictor(oos):
    """ME in the OOS table is raw log market cap: exp(ME) must be positive and
    span orders of magnitude, which a rank-standardised [-1, 1] column cannot.
    """
    cap = np.exp(oos["ME"].to_numpy())
    assert (cap > 0).all()
    assert cap.max() / cap.min() > 100


def test_published_headline_vw_mse_reproduces(oos):
    """Pins the README's table to the shipped artifact."""
    df = evaluate.load()
    monthly = evaluate.monthly_vw_mse(df)
    vw = monthly.mean()

    assert len(df) == 10_145
    assert monthly.shape[0] == 120
    assert vw["LSTM"] == pytest.approx(0.0995, abs=5e-5)
    assert vw["OLS_1Y_D"] == pytest.approx(0.0974, abs=5e-5)
    assert vw["OLS_5Y_M"] == pytest.approx(0.1968, abs=5e-5)


def test_published_dm_results_reproduce(oos):
    """Pins the horse race: a tie against 1Y daily, a decisive win over 5Y monthly."""
    df = evaluate.load()
    monthly = evaluate.monthly_vw_mse(df)

    _, dm_1yd, p_1yd, _ = dm_test.dm_stat(monthly["LSTM"] - monthly["OLS_1Y_D"])
    _, dm_5ym, p_5ym, _ = dm_test.dm_stat(monthly["LSTM"] - monthly["OLS_5Y_M"])

    assert dm_1yd == pytest.approx(0.20, abs=0.01) and p_1yd > 0.05
    assert dm_5ym == pytest.approx(-6.55, abs=0.01) and p_5ym < 0.001


def test_published_size_tercile_result_reproduces(oos):
    """Pins the README's size table: LSTM wins small and mid, loses large."""
    from betaforecast.importance import by_size

    df = by_size.assign_terciles(oos.copy())
    df["cap"] = np.exp(df["ME"])
    vw = by_size.group_mse(df, weighted=True)
    dm = by_size.group_dm(df, weighted=True)

    assert vw.loc["Small", "LSTM"] < vw.loc["Small", "OLS_1Y_D"]
    assert vw.loc["Mid", "LSTM"] < vw.loc["Mid", "OLS_1Y_D"]
    assert vw.loc["Large", "LSTM"] > vw.loc["Large", "OLS_1Y_D"]

    assert dm.loc["Small", "p_OLS_1Y_D"] < 0.05
    assert dm.loc["Mid", "p_OLS_1Y_D"] < 0.05
    assert dm.loc["Large", "p_OLS_1Y_D"] > 0.05

    # the 5Y-monthly benchmark is beaten in every tercile
    assert (dm["DM_OLS_5Y_M"] < 0).all()
    assert (dm["p_OLS_5Y_M"] < 0.001).all()


def test_readme_figures_regenerate_from_the_shipped_artifact(oos, tmp_path,
                                                             monkeypatch):
    """The README embeds these, so a clone must be able to rebuild them."""
    from betaforecast.figures import readme_figures as rf

    monkeypatch.setattr(rf, "DOCS_IMG", tmp_path)
    rf.report()

    for name in ("horse_race.png", "size_terciles.png"):
        assert (tmp_path / name).stat().st_size > 10_000
