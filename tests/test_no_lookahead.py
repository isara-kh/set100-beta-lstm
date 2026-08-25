"""Look-ahead invariants.

Nothing here checks that the pipeline runs; these assert that information from
the future cannot reach a forecast. They are the claims a reader would
otherwise have to take on trust.
"""
import numpy as np
import pytest

from betaforecast.evaluation import dm_test
from betaforecast.features import sequences as S
from betaforecast.features import standardize as Z

from conftest import make_panel

ITERATIONS = S.walk_forward_iterations()


@pytest.mark.parametrize("it", ITERATIONS, ids=lambda it: f"iter{it['iter']}")
def test_gap_year_separates_validation_from_test(it):
    """The discarded year sits strictly between validation and test."""
    assert it["gap_year"] == it["val_year"] + 1
    assert it["test_year"] == it["gap_year"] + 1


@pytest.mark.parametrize("it", ITERATIONS, ids=lambda it: f"iter{it['iter']}")
def test_gap_is_never_trained_or_evaluated_on(it):
    """The gap year appears in no split at all."""
    assert it["gap_year"] not in it["train_years"]
    assert it["gap_year"] != it["val_year"]
    assert it["gap_year"] != it["test_year"]


@pytest.mark.parametrize("it", ITERATIONS, ids=lambda it: f"iter{it['iter']}")
def test_splits_are_disjoint_and_ordered(it):
    """Train precedes validation precedes test, with no year in common."""
    train, val, test = set(it["train_years"]), it["val_year"], it["test_year"]
    assert train.isdisjoint({val, test})
    assert val != test
    assert max(train) < val < test


def test_gap_length_equals_the_forecast_horizon():
    """The buffer is one year because the target looks 12 months ahead.

    If the horizon ever changed without the gap following it, validation labels
    would overlap the test window again.
    """
    for it in ITERATIONS:
        gap_months = (it["test_year"] - it["val_year"] - 1) * 12
        assert gap_months == dm_test.H


def test_forecast_origin_must_be_a_member_month(panel):
    """A non-member origin is dropped even when its data is complete."""
    n_before = len(S.build_sequences(panel)[1])

    target = panel.index[(panel["ric"] == "AAA")][20]
    panel.loc[target, "is_member"] = False
    X, y, meta = S.build_sequences(panel)

    assert len(y) == n_before - 1
    dropped = panel.loc[target, "ym"]
    assert not ((meta["ric"] == "AAA") & (meta["ym"] == dropped)).any()


def test_interior_lookback_month_may_be_non_member(panel):
    """Membership is required at the origin only, not inside the window."""
    n_before = len(S.build_sequences(panel)[1])

    # 6 months before an origin that itself stays a member
    interior = panel.index[(panel["ric"] == "AAA")][14]
    panel.loc[interior, "is_member"] = False

    assert len(S.build_sequences(panel)[1]) == n_before - 1  # only its own origin


def test_calendar_gap_invalidates_every_window_spanning_it(panel):
    """A missing month kills the L origins whose lookback would span it."""
    n_before = len(S.build_sequences(panel)[1])

    missing = panel.index[(panel["ric"] == "AAA")][20]
    gapped = panel.drop(index=missing).reset_index(drop=True)
    X, y, meta = S.build_sequences(gapped)

    # the origin itself, plus the L-1 later origins that need it in-window
    assert len(y) == n_before - S.L
    assert (meta["ric"] == "AAA").sum() < (n_before // 2)


def test_dfy_scaling_is_fit_on_the_training_fold_alone():
    """The per-fold DFY min-max must not see validation or test rows."""
    panel = make_panel(n_months=240, start="2005-01", seed=7)
    X, y, meta = Z.build_model_inputs(panel)
    it = next(i for i in ITERATIONS
              if S.split_indices(meta, i)[2].any())          # a non-empty test fold
    train, _, test = S.split_indices(meta, it)

    fold = Z.fold_inputs(X, y, meta, it)

    assert fold["params"] == Z.fit_dfy(X[train])
    # and it is genuinely a subset: fitting on everything gives a wider range
    whole = Z.fit_dfy(X)
    assert whole["min"] <= fold["params"]["min"]
    assert whole["max"] >= fold["params"]["max"]


def test_test_fold_rows_are_all_later_than_training_rows():
    """No training origin may fall on or after a test origin."""
    panel = make_panel(n_months=240, start="2005-01", seed=3)
    X, y, meta = Z.build_model_inputs(panel)
    for it in ITERATIONS:
        train, val, test = S.split_indices(meta, it)
        if not (train.any() and test.any()):
            continue
        assert meta.loc[train, "ym"].max() < meta.loc[test, "ym"].min()
        if val.any():
            assert meta.loc[val, "ym"].max() < meta.loc[test, "ym"].min()
