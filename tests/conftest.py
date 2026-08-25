"""Synthetic panel fixtures.

The real panel is not distributed, so the sequence-validity tests build a small
panel by hand: it is the only way to assert what gets *rejected*, which is the
part that matters for look-ahead.
"""
import numpy as np
import pandas as pd
import pytest

from betaforecast.features import sequences as S


def make_panel(n_months=40, rics=("AAA", "BBB"), start="2010-01", seed=0):
    """A well-formed panel: contiguous months, every origin a member-month."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(start=start, periods=n_months, freq="M")
    frames = []
    for ric in rics:
        df = pd.DataFrame({"ric": ric, "ym": months})
        df["date"] = df["ym"].dt.to_timestamp(how="end").dt.normalize()
        for p in S.PREDICTORS:
            df[p] = rng.normal(size=n_months)
        df[S.TARGET] = rng.normal(size=n_months)
        df["is_member"] = True
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def panel():
    return make_panel()


@pytest.fixture
def oos():
    """The shipped out-of-sample forecast table, or skip if absent."""
    from betaforecast.paths import OOS_FORECASTS
    if not OOS_FORECASTS.exists():
        pytest.skip("oos_forecasts.parquet not present")
    return pd.read_parquet(OOS_FORECASTS)
