"""Subsample cuts of accuracy and importance: size tercile x COVID regime.

Slices the zero-out importance by size tercile and by pre/post-COVID regime, and
runs the LSTM-vs-benchmark DM test per tercile x regime cell. The muted
forecasts are cached to importance_cells.parquet so re-slicing skips the ~5 min
pass through the saved ensembles.
"""
import os

import numpy as np
import pandas as pd

from betaforecast.importance import zero_out as IMP
from betaforecast.importance import by_size as SS
from betaforecast.evaluation import dm_test
from betaforecast.features import standardize as Z
from betaforecast.paths import OOS_FORECASTS as OOS, DATA_PROCESSED, REPO

CELLS = DATA_PROCESSED / "importance_cells.parquet"
PREDICTORS = Z.PREDICTORS


def vw_mse(df, col):
    """Average of monthly value-weighted MSE of `col` within `df`.

    Weights are within-month market-cap shares over whatever rows `df` holds, so
    passing a tercile-month or regime slice gives that slice's VW-MSE directly.
    """
    w = df.groupby("ym")["cap"].transform(lambda s: s / s.sum())
    se = (df["y_true"] - df[col]) ** 2
    return float((w * se).groupby(df["ym"]).sum().mean())


def importance_within(df):
    """Zero-out VW-MSE gap per predictor over the rows of `df`."""
    base = vw_mse(df, "base")
    return {p: vw_mse(df, f"zero_{p}") - base for p in PREDICTORS}


def build_cells():
    """Per-stock-month muted forecasts + ME/y_true + tercile + regime, cached."""
    if os.path.exists(CELLS):
        return pd.read_parquet(CELLS)
    fc = IMP.collect_forecasts()
    oos = pd.read_parquet(OOS)[["ric", "ym", "ME", "y_true", "yhat_lstm"]]
    df = fc.merge(oos, on=["ric", "ym"], how="left", validate="one_to_one")
    diff = np.abs(df["base"] - df["yhat_lstm"]).max()
    assert diff < 1e-5, f"baseline mismatch (max |diff|={diff:.2e})"
    df["cap"] = np.exp(df["ME"])
    df = SS.assign_terciles(df)                       # adds size_grp
    df["regime"] = np.where(df["ym"] <= dm_test.PRE_END, "pre",
                            np.where(df["ym"] >= dm_test.COVID, "post", "drop"))
    df.to_parquet(CELLS)
    return df


def tercile_regime_dm(oos):
    """Diebold-Mariano test, LSTM vs each benchmark, per size group x regime (value-weighted)."""
    df = SS.assign_terciles(oos.copy())
    df["cap"] = np.exp(df["ME"])
    df["regime"] = np.where(df["ym"] <= dm_test.PRE_END, "pre",
                            np.where(df["ym"] >= dm_test.COVID, "post", "drop"))
    rows = []
    for g in SS.GROUPS:
        for r in ("pre", "post"):
            sub = df[(df["size_grp"] == g) & (df["regime"] == r)]
            w = sub.groupby("ym")["cap"].transform(lambda s: s / s.sum())
            se_l = (sub["y_true"] - sub["yhat_lstm"]) ** 2
            row = {"group": g, "regime": r}
            for name, col, _ in SS.MODELS[1:]:
                delta = se_l - (sub["y_true"] - sub[col]) ** 2
                d = (w * delta).groupby(sub["ym"]).sum().sort_index()
                _, dm, p, _ = dm_test.dm_stat(d)
                row[f"DM_{name}"], row[f"p_{name}"] = dm, p
                row[f"MSE_LSTM"] = vw_mse(sub, "yhat_lstm")
                row[f"MSE_{name}"] = vw_mse(sub, col)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["group", "regime"])


def report():
    df = build_cells()

    print("\n=== H1 add-on: zero-out importance by SIZE TERCILE (VW) ===")
    ter = pd.DataFrame({g: importance_within(df[df["size_grp"] == g])
                        for g in SS.GROUPS})
    print(ter.reindex(ter["Mid"].abs().sort_values(ascending=False).index)
          .round(4).to_string())

    print("\n=== H2 add-on: zero-out importance by REGIME (VW) ===")
    reg = pd.DataFrame({r: importance_within(df[df["regime"] == r])
                        for r in ("pre", "post")})
    print(reg.reindex(reg["pre"].abs().sort_values(ascending=False).index)
          .round(4).to_string())

    print("\n=== H2 add-on: tercile x regime DM accuracy (VW) ===")
    trd = tercile_regime_dm(pd.read_parquet(OOS))
    print(trd.round(3).to_string())

    ter.to_parquet(os.path.join(REPO, "data", "processed", "importance_by_tercile.parquet"))
    reg.to_parquet(os.path.join(REPO, "data", "processed", "importance_by_regime.parquet"))
    trd.to_parquet(os.path.join(REPO, "data", "processed", "dm_tercile_regime.parquet"))
    print("\nSaved: importance_by_tercile / importance_by_regime / dm_tercile_regime .parquet")


if __name__ == "__main__":
    report()
