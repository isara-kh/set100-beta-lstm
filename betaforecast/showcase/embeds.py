"""One writer function per Live Embed; each returns the JSON-ready payload."""
import pandas as pd

from betaforecast.evaluation import dm_test, evaluate
from betaforecast.features import sequences as seq
from betaforecast.importance import subsamples
from betaforecast.paths import PREDICTOR_IMPORTANCE_CSV, rel

EMBEDS = {}

# Rounding at export (decimal places).
DP_DIFF, DP_MSE, DP_T, DP_P = 6, 4, 2, 3
# Zero-out importance (VI) is a difference of value-weighted MSEs, so it takes
# the differential precision, not DP_MSE: VI runs ~0.001-0.014 and 4 dp would
# leave only 1-2 significant digits.
DP_VI = DP_DIFF


def embed(name):
    """Register a writer ``fn(df) -> dict`` under the embed id ``name``."""
    def register(fn):
        EMBEDS[name] = fn
        return fn
    return register


def num(x, dp):
    """Round to ``dp`` places as a plain float; folds -0.0 into 0.0."""
    return round(float(x), dp) + 0.0


def month_label(ym):
    return str(ym)[:7]


@embed("walk-forward")
def walk_forward(df):
    return {
        "start_year": seq.FIRST_TRAIN_START,
        "train_years": seq.TRAIN_YEARS,
        "validation_years": seq.VALIDATION_YEARS,
        "gap_years": seq.GAP_YEARS,
        "test_years": seq.TEST_YEARS,
        "iterations": seq.N_ITERATIONS,
    }


@embed("horse-race")
def horse_race(df):
    monthly = evaluate.monthly_vw_mse(df).sort_index()
    keys = {"OLS_1Y_D": "ols_1y_d", "OLS_5Y_M": "ols_5y_m"}
    diffs = {b: monthly["LSTM"] - monthly[b] for b in keys}
    months = []
    for ym in monthly.index:
        row = {"origin": month_label(ym)}
        for b, k in keys.items():
            row[f"d_vs_{k}"] = num(diffs[b].loc[ym], DP_DIFF)
        months.append(row)
    dm = {}
    for b, k in keys.items():
        dbar, t, p, _ = dm_test.dm_stat(diffs[b])
        dm[k] = {"dbar": num(dbar, DP_DIFF), "t": num(t, DP_T), "p": num(p, DP_P)}
    return {"covid_start": dm_test.COVID, "months": months, "dm": dm}


@embed("importance")
def importance(df):
    # Tests point PREDICTOR_IMPORTANCE_CSV (this module's name) at a temp path.
    path = PREDICTOR_IMPORTANCE_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {rel(path)}, the committed predictor-importance table. "
            "Produce it locally with `python -m betaforecast.importance.zero_out` "
            "(needs the trained models, which are not distributed), then commit it.")
    tab = pd.read_csv(path).sort_values("VI", ascending=False, kind="stable")
    return {"predictors": [
        {"name": r.predictor, "vi": num(r.VI, DP_VI),
         "t": num(r.t_stat, DP_T), "p": num(r.p_value, DP_P)}
        for r in tab.itertuples()]}


@embed("where-it-wins")
def where_it_wins(df):
    """24 cells: weighting x regime x size group, in that (deterministic) order.

    EW is the mean of monthly means (weights 1/N_t per origin month), for the
    MSE and the DM differential alike; see ``subsamples.monthly_loss``.
    """
    cells = []
    by_weighting = {w: subsamples.group_regime_cells(df, weighted=(w == "vw"))
                    for w in ("vw", "ew")}
    for w in ("vw", "ew"):
        for r in subsamples.REGIMES:
            for g in subsamples.GROUPS_ALL:
                c = by_weighting[w][(g, r)]
                cells.append({
                    "weighting": w, "regime": r, "group": g,
                    "mse": {"lstm": num(c["MSE_LSTM"], DP_MSE),
                            "ols_1y_d": num(c["OLS_1Y_D"]["MSE"], DP_MSE),
                            "ols_5y_m": num(c["OLS_5Y_M"]["MSE"], DP_MSE)},
                    "dm": {"ols_1y_d": {"t": num(c["OLS_1Y_D"]["DM"], DP_T),
                                        "p": num(c["OLS_1Y_D"]["p"], DP_P)},
                           "ols_5y_m": {"t": num(c["OLS_5Y_M"]["DM"], DP_T),
                                        "p": num(c["OLS_5Y_M"]["p"], DP_P)}}})
    return {"cells": cells}
