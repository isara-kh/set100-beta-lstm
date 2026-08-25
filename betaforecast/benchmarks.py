"""Align the two rolling-OLS benchmarks to the LSTM's out-of-sample set.

The benchmarks are persistence forecasts -- the rolling historical beta at the
origin is used as-is as next year's forecast: OLS_1Y_D (one year of daily
returns) and OLS_5Y_M (five years of monthly returns), both precomputed in the
data step. This pulls their raw values at each OOS (ric, ym), carries raw ME
along for the value weights, and writes the unified table oos_forecasts.parquet.
"""
import os

import numpy as np
import pandas as pd

from betaforecast.paths import (LSTM_OOS_FORECASTS as LSTM_OOS,
                                LSTM_INPUT_PANEL as PANEL,
                                OOS_FORECASTS as OUT, REPO)

BENCHMARKS = {"OLS_5Y_M": "yhat_ols5ym", "OLS_1Y_D": "yhat_ols1yd"}


def build_benchmark_forecasts():
    """Align the two persistence benchmarks to the LSTM OOS set; save the table.

    Returns the unified forecast table with one row per OOS (ric, ym):
    realized target, the LSTM forecast, the two benchmark forecasts, and raw ME.
    """
    oos = pd.read_parquet(LSTM_OOS)
    cols = ["ric", "ym", "ME"] + list(BENCHMARKS)
    panel = pd.read_parquet(PANEL)[cols]

    m = oos.merge(panel, on=["ric", "ym"], how="left", validate="one_to_one")
    if m[list(BENCHMARKS)].isna().any().any():
        raise ValueError("benchmark values missing for some OOS origins")
    m = m.rename(columns=BENCHMARKS)

    out = m[["ric", "ym", "date", "iter", "ME", "y_true",
             "yhat_lstm", "yhat_ols5ym", "yhat_ols1yd"]]
    out.to_parquet(OUT)
    return out


def _ew_mse(out, col):
    return float(np.mean((out["y_true"] - out[col]) ** 2))


def report():
    out = build_benchmark_forecasts()
    print(f"Unified OOS forecast table: {len(out):,} rows "
          f"({out['ym'].min()} -> {out['ym'].max()}), "
          f"{out['ric'].nunique()} stocks -> {os.path.relpath(OUT, REPO)}\n")
    print(f"{'model':14s}{'forecast col':>14s}{'pooled EW-MSE':>15s}{'mean fcast':>12s}")
    rows = [("LSTM", "yhat_lstm"),
            ("OLS_5Y_M", "yhat_ols5ym"),
            ("OLS_1Y_D", "yhat_ols1yd")]
    for name, col in rows:
        print(f"{name:14s}{col:>14s}{_ew_mse(out, col):>15.4f}{out[col].mean():>12.3f}")
    print(f"\n(realized target mean = {out['y_true'].mean():.3f}; "
          "pooled equal-weighted MSE is a first look only -- the primary metric "
          "is\nthe value-weighted MSE, computed next.)")


if __name__ == "__main__":
    report()
