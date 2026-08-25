"""Non-beta firm and market predictors used alongside historical betas.

Each helper returns a monthly (ric, year-month) series for one predictor:
    ME     - firm size, ln(market cap in million baht)
    DIVPAY - dividend-payout ratio
    TO     - turnover / trading value, ln(monthly baht value traded)
    MOM    - momentum (built elsewhere from past returns)
    IVOL   - idiosyncratic volatility, ln(std of daily Fama-French 3-factor
             regression residuals within the month)
    ILLIQ  - Amihud illiquidity, |monthly return| / trading value
    DFY    - default yield spread (a market-wide series)
"""
import numpy as np
import pandas as pd

from betaforecast.data import sources as data_io

MIN_IVOL_D = 15  # min daily obs for the monthly FF3 IVOL regression


def trading_value_monthly(stock_daily):
    """Monthly (ric, ym): sum of daily trading value (million baht) and |sum raw r|.

    Trading value = adjusted close x share volume / 1e6.  Returns columns
    TOval (sum value), absR (|sum raw daily log return|).
    """
    prices = data_io.load_prices()[["ric", "date", "close"]]
    vol = data_io.load_volume()[["ric", "date", "volume"]]
    pv = prices.merge(vol, on=["ric", "date"], how="inner")
    pv["tv"] = pv["close"] * pv["volume"].astype("float64") / 1e6  # million baht
    pv["ym"] = pv["date"].dt.to_period("M")

    raw = stock_daily[["ric", "date", "r"]].copy()
    raw["ym"] = raw["date"].dt.to_period("M")
    sumr = raw.groupby(["ric", "ym"])["r"].sum().rename("sumR")

    g = pv.groupby(["ric", "ym"]).agg(TOval=("tv", "sum")).join(sumr)
    g["absR"] = g["sumR"].abs()
    return g


def ivol_monthly(stock_daily):
    """Monthly (ric, ym) IVOL = ln(std of daily FF3 residuals), via group sufficient stats."""
    fac = data_io.load_factors()[["MKT", "SMB", "HML"]]
    df = stock_daily[["ric", "date", "r_excess"]].merge(
        fac, left_on="date", right_index=True, how="inner").dropna()
    df["ym"] = df["date"].dt.to_period("M")

    y, f1, f2, f3 = df["r_excess"], df["MKT"], df["SMB"], df["HML"]
    cols = {
        "n": 1.0, "Sy": y, "S1": f1, "S2": f2, "S3": f3,
        "yy": y * y, "y1": y * f1, "y2": y * f2, "y3": y * f3,
        "11": f1 * f1, "12": f1 * f2, "13": f1 * f3,
        "22": f2 * f2, "23": f2 * f3, "33": f3 * f3,
    }
    agg = pd.DataFrame(cols).assign(ric=df["ric"].values, ym=df["ym"].values)
    s = agg.groupby(["ric", "ym"]).sum()
    s = s[s["n"] >= MIN_IVOL_D]

    out = np.full(len(s), np.nan)
    S = s.to_dict("records")
    for i, r in enumerate(S):
        n = r["n"]
        XtX = np.array([
            [n,      r["S1"], r["S2"], r["S3"]],
            [r["S1"], r["11"], r["12"], r["13"]],
            [r["S2"], r["12"], r["22"], r["23"]],
            [r["S3"], r["13"], r["23"], r["33"]],
        ])
        Xty = np.array([r["Sy"], r["y1"], r["y2"], r["y3"]])
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue
        rss = r["yy"] - beta @ Xty
        if rss <= 0 or n <= 4:
            continue
        out[i] = np.log(np.sqrt(rss / (n - 1)))  # residuals sum to 0 (intercept)
    return pd.Series(out, index=s.index, name="IVOL")


def me_monthly():
    """Month-end ME = ln(market cap in million baht), last daily cap within the month."""
    mc = data_io.load_market_cap()
    mc["ym"] = mc["date"].dt.to_period("M")
    last = mc.sort_values("date").groupby(["ric", "ym"]).tail(1)
    s = last.set_index(["ric", "ym"])["market_cap"] / 1e6  # baht -> million baht
    return np.log(s).rename("ME")


def divpay_monthly():
    d = data_io.load_divpay()
    d["ym"] = d["date"].dt.to_period("M")
    return d.set_index(["ric", "ym"])["divpay"].rename("DIVPAY")


def dfy_monthly():
    d = data_io.load_dfy().copy()
    d["ym"] = d.index.to_period("M")
    return d.groupby("ym")["DFY"].last().rename("DFY")
