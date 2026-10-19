"""Subsample cuts of accuracy and importance: size tercile x COVID regime.

Two halves with different input requirements, kept separately runnable:

* `accuracy_report` -- the DM test per tercile x regime cell. Needs only the
  published oos_forecasts.parquet, so it runs from a bare clone.
* `importance_report` -- the zero-out importance slices. Needs the input panel
  and the trained ensembles, neither of which is distributed. The muted
  forecasts are cached to importance_cells.parquet so re-slicing skips the
  ~5 min pass through the saved models.

`python -m betaforecast.importance.subsamples --dm-only` runs the first alone.
"""
import argparse
import os

import numpy as np
import pandas as pd

from betaforecast.importance import by_size as SS
from betaforecast.evaluation import dm_test
from betaforecast.features import standardize as Z
from betaforecast.paths import (OOS_FORECASTS as OOS, LSTM_INPUT_PANEL as PANEL,
                                MODELS as MODEL_DIR, DATA_PROCESSED, rel)

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
    # Imported here, not at module scope: it pulls in keras, which the DM-only
    # path has no use for.
    from betaforecast.importance import zero_out as IMP
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


REGIMES = ("all", "pre", "post")
GROUPS_ALL = ("all", *(g.lower() for g in SS.GROUPS))


def with_cuts(oos):
    """OOS table plus cap, size tercile and regime, all cut on the origin."""
    df = SS.assign_terciles(oos.copy())
    df["cap"] = np.exp(df["ME"])
    df["regime"] = np.where(df["ym"] <= dm_test.PRE_END, "pre",
                            np.where(df["ym"] >= dm_test.COVID, "post", "drop"))
    return df


def monthly_loss(sub, se, weighted):
    """Per-origin-month loss series of squared errors `se` over the rows of `sub`.

    Value-weighted: within-month market-cap shares. Equal-weighted: weights
    1/N_t, so the monthly loss is that month's mean squared error. Every
    reported EW figure -- the MSE and the DM loss differential -- is the average
    (or DM test) of this one monthly series, i.e. the *mean of monthly means*,
    as the manuscript defines EW ("replaces w_i,t with 1/N_t", then averages
    over the origins). It is not the pooled stock-month mean.
    """
    if weighted:
        w = sub.groupby("ym")["cap"].transform(lambda s: s / s.sum())
        return (w * se).groupby(sub["ym"]).sum().sort_index()
    return se.groupby(sub["ym"]).mean().sort_index()


def mse_within(sub, col, weighted):
    """Average over origin months of the monthly (VW or EW) MSE of `col`."""
    return float(monthly_loss(sub, (sub["y_true"] - sub[col]) ** 2, weighted).mean())


def cell_stats(sub, weighted):
    """MSEs and DM (LSTM vs each benchmark) for one slice of the OOS table."""
    se_l = (sub["y_true"] - sub["yhat_lstm"]) ** 2
    out = {"MSE_LSTM": mse_within(sub, "yhat_lstm", weighted)}
    for name, col, _ in SS.MODELS[1:]:
        delta = se_l - (sub["y_true"] - sub[col]) ** 2
        _, dm, p, _ = dm_test.dm_stat(monthly_loss(sub, delta, weighted))
        out[name] = {"DM": dm, "p": p, "MSE": mse_within(sub, col, weighted)}
    return out


def group_regime_cells(oos, weighted=True, groups=GROUPS_ALL, regimes=REGIMES):
    """Per size group x regime cell: MSEs and Diebold-Mariano, LSTM vs each
    benchmark, value-weighted (`weighted=True`) or equal-weighted.

    Group "all" is every stock; "small"/"mid"/"large" are the origin-month
    terciles. Regime "all" is all 120 origins; "pre"/"post" drop the two crisis
    origins. Returns {(group, regime): cell_stats}, in loop order.
    """
    df = with_cuts(oos)
    cells = {}
    for g in groups:
        in_group = df if g == "all" else df[df["size_grp"] == g.capitalize()]
        for r in regimes:
            sub = in_group if r == "all" else in_group[in_group["regime"] == r]
            cells[(g, r)] = cell_stats(sub, weighted)
    return cells


def tercile_regime_dm(oos, weighted=True):
    """Diebold-Mariano test, LSTM vs each benchmark, per size group x regime
    (value-weighted by default); the accuracy report's table."""
    terciles = [g.lower() for g in SS.GROUPS]
    cells = group_regime_cells(oos, weighted, terciles, ("pre", "post"))
    rows = []
    for (g, r), c in cells.items():
        row = {"group": g.capitalize(), "regime": r}
        for name, _, _ in SS.MODELS[1:]:
            row[f"DM_{name}"], row[f"p_{name}"] = c[name]["DM"], c[name]["p"]
            row["MSE_LSTM"] = c["MSE_LSTM"]
            row[f"MSE_{name}"] = c[name]["MSE"]
        rows.append(row)
    return pd.DataFrame(rows).set_index(["group", "regime"])


def accuracy_report():
    """DM accuracy per tercile x regime. Needs only the published OOS table."""
    print("\n=== Tercile x regime DM accuracy (VW) ===")
    trd = tercile_regime_dm(pd.read_parquet(OOS))
    print(trd.round(3).to_string())
    out = DATA_PROCESSED / "dm_tercile_regime.parquet"
    trd.to_parquet(out)
    print(f"\nSaved: {rel(out)}")
    return trd


def importance_report():
    """Zero-out importance sliced by tercile and by regime. Needs the input
    panel and the trained ensembles, so it cannot run from a bare clone.
    """
    df = build_cells()

    print("\n=== Zero-out importance by SIZE TERCILE (VW) ===")
    ter = pd.DataFrame({g: importance_within(df[df["size_grp"] == g])
                        for g in SS.GROUPS})
    print(ter.reindex(ter["Mid"].abs().sort_values(ascending=False).index)
          .round(4).to_string())

    print("\n=== Zero-out importance by REGIME (VW) ===")
    reg = pd.DataFrame({r: importance_within(df[df["regime"] == r])
                        for r in ("pre", "post")})
    print(reg.reindex(reg["pre"].abs().sort_values(ascending=False).index)
          .round(4).to_string())

    ter.to_parquet(DATA_PROCESSED / "importance_by_tercile.parquet")
    reg.to_parquet(DATA_PROCESSED / "importance_by_regime.parquet")
    print("\nSaved: importance_by_tercile / importance_by_regime .parquet")
    return ter, reg


def report(dm_only=False):
    """Accuracy first (always available), then importance when its inputs are."""
    accuracy_report()
    if dm_only:
        return
    missing = [rel(p) for p in (PANEL, MODEL_DIR) if not os.path.exists(p)]
    if missing and not os.path.exists(CELLS):
        print(f"\nSkipping zero-out importance: {', '.join(missing)} not present "
              "(not distributed -- see the data availability note in README).")
        return
    importance_report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dm-only", action="store_true",
                    help="run only the tercile x regime DM test (no models needed)")
    report(dm_only=ap.parse_args().dm_only)
