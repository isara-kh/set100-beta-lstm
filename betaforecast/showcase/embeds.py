"""One writer function per Live Embed; each returns the JSON-ready payload."""
from betaforecast.evaluation import dm_test, evaluate
from betaforecast.features import sequences as seq

EMBEDS = {}

# Rounding at export (decimal places).
DP_DIFF, DP_MSE, DP_T, DP_P = 6, 4, 2, 3


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
