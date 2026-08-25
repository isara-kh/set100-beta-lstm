"""Out-of-sample evaluation: the value-weighted MSE, the study's primary metric.

Within each origin month, each stock's squared error is weighted by its
market-cap share (from RAW ME, never the rank-standardised predictor); the
full-sample VW-MSE is the simple average of the 120 monthly figures. Also writes
the monthly series (monthly_vw_mse.parquet) that the Diebold-Mariano test uses.
"""
import os

import numpy as np
import pandas as pd

from betaforecast.paths import OOS_FORECASTS as OOS, MONTHLY_VW_MSE as MONTHLY_OUT, REPO

MODELS = {"LSTM": "yhat_lstm", "OLS_5Y_M": "yhat_ols5ym", "OLS_1Y_D": "yhat_ols1yd"}


def load():
    """Load the unified OOS table and add market cap + monthly value weights."""
    df = pd.read_parquet(OOS)
    df["mktcap"] = np.exp(df["ME"])                 # ME = ln(market cap, mn baht)
    df["w"] = df.groupby("ym")["mktcap"].transform(lambda s: s / s.sum())
    return df


def monthly_vw_mse(df):
    """Per model, the monthly value-weighted MSE series indexed by origin month.

    Returns a DataFrame (rows = 120 OOS months, cols = model names) of
    sum_i w_{i,t} (y - yhat)^2 within each month.
    """
    out = {}
    for name, col in MODELS.items():
        se = (df["y_true"] - df[col]) ** 2
        out[name] = (df["w"] * se).groupby(df["ym"]).sum()
    m = pd.DataFrame(out)
    m.index.name = "ym"
    return m


def summary(df, monthly):
    """Full-sample VW-MSE (simple average of the 120 monthly VW-MSEs) per model."""
    rows = []
    for name in MODELS:
        rows.append({"model": name, "VW_MSE": monthly[name].mean()})
    return pd.DataFrame(rows).set_index("model")


def report():
    df = load()
    monthly = monthly_vw_mse(df)
    monthly.to_parquet(MONTHLY_OUT)
    s = summary(df, monthly)

    n_months = monthly.shape[0]
    print(f"OOS: {len(df):,} stock-months over {n_months} month-ends "
          f"({df['ym'].min()} -> {df['ym'].max()})\n")

    print(f"{'model':10s}{'VW-MSE':>10s}{'vs LSTM (VW)':>14s}")
    base = s.loc["LSTM", "VW_MSE"]
    for name in MODELS:
        vw = s.loc[name, "VW_MSE"]
        rel = "" if name == "LSTM" else f"{(vw - base):+.4f}"
        print(f"{name:10s}{vw:>10.4f}{rel:>14s}")

    print("\nPrimary metric = VW-MSE (simple average of the 120 monthly "
          "value-weighted\nMSEs). 'vs LSTM (VW)' is the benchmark's VW-MSE minus "
          "the LSTM's: positive\nmeans the LSTM is more accurate. Monthly VW "
          f"series saved -> {os.path.relpath(MONTHLY_OUT, REPO)}\n"
          "(feeds the Diebold-Mariano test).")


if __name__ == "__main__":
    report()
