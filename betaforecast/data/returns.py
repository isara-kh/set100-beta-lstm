"""Daily and monthly return construction.

Turns raw daily prices, dividends and risk-free levels into the return series
the beta panel needs: per-stock daily total and excess returns, plus daily and
monthly market excess returns. All returns are continuously compounded (log)
returns unless noted otherwise.
"""
import numpy as np
import pandas as pd

from betaforecast.data import dividends, sources as data_io


def build_daily_returns(total_return=True):
    """Return (stock_daily, mkt_excess, rf, raw_mkt).

    stock_daily : long frame [ric, date, r_price, div_yield, r, rf, r_excess].
                  ``r`` is the daily TOTAL return (price log return + cash-dividend
                  yield on ex-dates) when total_return=True, else the price return.
    mkt_excess  : daily market EXCESS return (the SET100 TRI is already total return).
    rf          : daily risk-free log return.   raw_mkt : daily raw market log return.
    """
    prices = data_io.load_prices()
    rf = data_io.load_rf_return()
    raw_mkt = data_io.load_market_return()

    # Daily stock split-adjusted price log returns, gap-safe within each ric.
    prices = prices.copy()
    prices["r_price"] = prices.groupby("ric")["close"].transform(lambda s: np.log(s / s.shift(1)))
    stock = prices.dropna(subset=["r_price"])[["ric", "date", "r_price"]].copy()

    # Cash-dividend yield per (ric, ex-trading-day) -> total return.
    if total_return:
        dates_by_ric = {r: pd.DatetimeIndex(g["date"].values)
                        for r, g in prices.groupby("ric")}
        close_by_ric = {r: g.set_index("date")["close"]
                        for r, g in prices.groupby("ric")}
        dy = dividends.dividend_yield_daily(dates_by_ric, close_by_ric)
        stock = stock.merge(dy, on=["ric", "date"], how="left")
        stock["div_yield"] = stock["div_yield"].fillna(0.0)
    else:
        stock["div_yield"] = 0.0
    stock["r"] = stock["r_price"] + stock["div_yield"]

    stock["rf"] = rf.reindex(stock["date"].values).values
    stock["r_excess"] = stock["r"] - stock["rf"]

    mkt = pd.concat([raw_mkt, rf], axis=1, join="inner")
    mkt_excess = (mkt["r_mkt"] - mkt["rf"]).rename("r_mkt_excess")
    return stock, mkt_excess, rf, raw_mkt


def month_end_index(dates):
    """Sorted list of last available trading day per calendar month."""
    s = pd.Series(1, index=pd.DatetimeIndex(sorted(pd.unique(dates))))
    return s.groupby(s.index.to_period("M")).apply(lambda x: x.index.max()).values


def build_monthly_excess(stock_daily, mkt_excess):
    """Monthly excess return = sum of daily excess within calendar month.

    Returns (stock_monthly, mkt_monthly) where the month label is the last
    trading day of the calendar month (month-end t).
    """
    sd = stock_daily.copy()
    sd["ym"] = sd["date"].dt.to_period("M")
    me = sd.groupby("ym")["date"].max()  # month-end trading day per period
    g = sd.groupby(["ric", "ym"])["r_excess"].sum().rename("r_excess_m").reset_index()
    g["date"] = g["ym"].map(me)
    stock_monthly = g[["ric", "date", "ym", "r_excess_m"]]

    mk = mkt_excess.to_frame("r_excess")
    mk["ym"] = mk.index.to_period("M")
    mkt_monthly = mk.groupby("ym")["r_excess"].sum().rename("r_mkt_excess_m")
    return stock_monthly, mkt_monthly
