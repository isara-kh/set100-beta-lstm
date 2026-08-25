"""Loading and cleaning of raw SET100 inputs.

All functions return tidy long/wide frames with a daily DatetimeIndex or a
(date, ric) layout. Prices are total-return-adjusted closes from Refinitiv.
"""
import os

import numpy as np
import pandas as pd

from betaforecast.paths import DATA_RAW as RAW


def _read_thai_csv(fname, value_col):
    """Read a dd/mm/YYYY, comma-thousand CSV (TRI, risk-free) into a sorted Series."""
    df = pd.read_csv(os.path.join(RAW, fname))
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Close"] = (
        df["Close"].astype(str).str.replace(",", "", regex=False).replace("nan", np.nan).astype(float)
    )
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    return df.set_index("Date")["Close"].rename(value_col)


def load_prices():
    """Long daily adjusted close: columns ric, date, close."""
    return pd.read_parquet(os.path.join(RAW, "prices_daily.parquet")).sort_values(["ric", "date"])


def load_volume():
    """Long daily share volume: columns ric, date, volume."""
    return pd.read_parquet(os.path.join(RAW, "trading_volume_daily.parquet")).sort_values(["ric", "date"])


def load_market_cap():
    """Long daily market cap (baht): columns ric, date, market_cap."""
    return pd.read_parquet(os.path.join(RAW, "market_cap_monthly.parquet")).sort_values(["ric", "date"])


def load_divpay():
    """Long month-end dividend-payout ratio: columns date, ric, divpay."""
    return pd.read_parquet(os.path.join(RAW, "divpay_monthly.parquet")).sort_values(["ric", "date"])


def load_membership():
    """Daily SET100 membership snapshot: columns date, ric, name."""
    return pd.read_parquet(os.path.join(RAW, "set100_membership.parquet"))


def load_factors():
    """Daily Thai FF/q factors indexed by date; MKT is already market excess return."""
    df = pd.read_csv(os.path.join(RAW, "qfactors_thai_daily.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")


def load_dfy():
    """Monthly default spread; index is month start as published by FRED."""
    df = pd.read_csv(os.path.join(RAW, "fred_dfy_monthly.csv"), parse_dates=["date"])
    return df.sort_values("date").set_index("date")[["BAA", "AAA", "DFY"]]


def load_market_return():
    """Daily SET100 TRI log return (raw, continuously compounded).

    The raw TRI file contains a few level prints on days the SET was closed
    (e.g. 2012-04-09, 2012-06-09, 2022-04-13); each phantom print is followed by
    a self-reversing partner return on the next trading day. Levels are filtered
    to equity trading days (dates present in prices_daily) before differencing,
    so the return spans the gap correctly.
    """
    lvl = _read_thai_csv("set100_tri_daily.csv", "tri")
    trading_days = pd.DatetimeIndex(load_prices()["date"].unique())
    lvl = lvl[lvl.index.isin(trading_days)]
    return np.log(lvl / lvl.shift(1)).dropna().rename("r_mkt")


def load_rf_return():
    """Daily risk-free log return from the ThaiBMA 1-3Y total-return index level."""
    lvl = _read_thai_csv("risk_free_daily.csv", "rf_index")
    return np.log(lvl / lvl.shift(1)).dropna().rename("rf")
