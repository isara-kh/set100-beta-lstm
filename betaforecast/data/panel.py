"""Month-end SET100 panel: the realized-beta target + 10 predictors per stock-month.

Design: everything reduces to monthly sums of daily moments on a (ric x ym)
grid; an OLS slope is a ratio of moment sums, so betas over any trailing or
forward window are rolling sums divided.
"""
import numpy as np
import pandas as pd

from betaforecast.data import sources as data_io, returns as R, predictors as pe

# Minimum-observation thresholds: a window is only used if it has at least this
# many observations, so betas/predictors are not estimated on sparse data.
MIN_3M_D = 40      # >= 40 daily obs in trailing 3 calendar months
MIN_1Y_D = 120     # >= 120 daily obs in trailing 12 calendar months
MIN_5Y_M = 36      # >= 36 monthly obs in trailing 60 calendar months
MIN_FWD_D = 200    # >= 200 daily obs in forward 12 months for a valid target
MIN_IVOL_D = 15    # >= 15 daily obs in the month for the FF3 IVOL regression


def _ols_beta(Sxy, Sxx, Sx, Sy, n):
    """OLS slope with intercept from window moment sums (vectorised)."""
    num = Sxy - Sx * Sy / n
    den = Sxx - Sx * Sx / n
    return num / den


def build():
    stock_daily, mkt_excess, rf, raw_mkt = R.build_daily_returns()

    # --- Daily merged frame: stock excess (y) and market excess (x) by date ---
    me = mkt_excess.rename("x").to_frame()
    d = stock_daily.merge(me, left_on="date", right_index=True, how="inner")
    d["y"] = d["r_excess"]
    d["ym"] = d["date"].dt.to_period("M")
    d["xy"] = d["x"] * d["y"]
    d["xx"] = d["x"] * d["x"]

    # Monthly sums of daily moments per (ric, ym).
    gm = d.groupby(["ric", "ym"]).agg(
        Sxy=("xy", "sum"), Sxx=("xx", "sum"),
        Sx=("x", "sum"), Sy=("y", "sum"), n=("y", "size"),
    ).reset_index()

    # Market-only monthly moments (same across stocks): forward target denominator.
    md = me.copy()
    md["ym"] = md.index.to_period("M")
    md["xx"] = md["x"] ** 2
    mkt_m = md.groupby("ym").agg(MSxx=("xx", "sum"), Mn=("x", "size"))

    # Full (ric x ym) grid spanning each ric's observed months.
    all_ym = pd.period_range(d["ym"].min(), d["ym"].max(), freq="M")
    panel = (
        gm.set_index(["ric", "ym"])
        .reindex(pd.MultiIndex.from_product([gm["ric"].unique(), all_ym], names=["ric", "ym"]))
    )

    moments = ["Sxy", "Sxx", "Sx", "Sy", "n"]

    def trailing_sum(col, k):
        return panel[col].groupby(level="ric").transform(
            lambda s: s.rolling(k, min_periods=1).sum())

    # ---- Predictors 1-2: trailing daily OLS betas (3M=3 months, 1Y=12 months) ----
    out = {}
    for name, k, minobs in [("OLS_3M_D", 3, MIN_3M_D), ("OLS_1Y_D", 12, MIN_1Y_D)]:
        S = {c: trailing_sum(c, k) for c in moments}
        beta = _ols_beta(S["Sxy"], S["Sxx"], S["Sx"], S["Sy"], S["n"])
        out[name] = beta.where(S["n"] >= minobs)

    # ---- Predictor 3: trailing 5Y monthly OLS beta ----
    # Monthly excess returns (stock & market) on the same grid.
    stock_m = (panel["Sy"]).rename("ry")           # Sy over a single month = monthly excess sum
    mkt_mexc = md.groupby("ym")["x"].sum().rename("rx")
    mm = pd.DataFrame({"ry": stock_m})
    mm["rx"] = mm.index.get_level_values("ym").map(mkt_mexc)
    mm["has"] = panel["n"].notna() & mm["ry"].notna()
    mm["xy"] = np.where(mm["has"], mm["rx"] * mm["ry"], np.nan)
    mm["xx"] = np.where(mm["has"], mm["rx"] * mm["rx"], np.nan)
    mm["x"] = np.where(mm["has"], mm["rx"], np.nan)
    mm["y"] = np.where(mm["has"], mm["ry"], np.nan)
    mm["one"] = np.where(mm["has"], 1.0, np.nan)

    def tsum_m(col, k=60):
        return mm[col].groupby(level="ric").transform(
            lambda s: s.rolling(k, min_periods=1).sum())

    Sxy = tsum_m("xy"); Sxx = tsum_m("xx"); Sx = tsum_m("x"); Sy = tsum_m("y"); nM = tsum_m("one")
    out["OLS_5Y_M"] = _ols_beta(Sxy, Sxx, Sx, Sy, nM).where(nM >= MIN_5Y_M)

    # ---- Target: forward 12-month uncentered realized beta over (t, t+12] ----
    # Sum of the next 12 months (excluding month t); requires a full 12-month
    # market window ahead.
    def forward12(s):
        fwd_incl = s[::-1].rolling(12, min_periods=12).sum()[::-1]  # months t..t+11
        return fwd_incl.shift(-1)                                   # months t+1..t+12

    fSxy = panel["Sxy"].groupby(level="ric", group_keys=False).apply(forward12)
    fN = panel["n"].groupby(level="ric", group_keys=False).apply(forward12)

    mkt_fSxx = forward12(mkt_m["MSxx"])
    mkt_fwd_full = forward12(mkt_m["Mn"])  # market trading days available ahead
    mkt_fSxx_map = pd.Series(panel.index.get_level_values("ym").map(mkt_fSxx), index=panel.index)
    mkt_full_map = pd.Series(panel.index.get_level_values("ym").map(mkt_fwd_full), index=panel.index)

    target = fSxy / mkt_fSxx_map
    valid_target = (fN >= MIN_FWD_D) & mkt_full_map.notna()
    out["beta_realized_fwd"] = target.where(valid_target)

    # ---- Predictor 7: MOM = sum of monthly excess returns over t-12..t-1 ----
    monthly_exc = panel["Sy"]  # single-month excess sum (NaN if no obs that month)
    mom = monthly_exc.groupby(level="ric").transform(
        lambda s: s.rolling(12, min_periods=12).sum().shift(1))
    out["MOM"] = mom

    res = pd.DataFrame(out)

    # ---- Predictors 4,5,6,8,9,10 merged on (ric, ym) ----
    tv = pe.trading_value_monthly(stock_daily)
    res["TO"] = np.log(tv["TOval"]).where(tv["TOval"] > 0)
    res["ILLIQ"] = (tv["absR"] / tv["TOval"]).where(tv["TOval"] > 0)
    res["IVOL"] = pe.ivol_monthly(stock_daily)
    res["ME"] = pe.me_monthly()
    res["DIVPAY"] = pe.divpay_monthly()
    dfy = pe.dfy_monthly()
    res["DFY"] = res.index.get_level_values("ym").map(dfy)

    res["n_daily_month"] = panel["n"]

    # Month-end trading day for each ym (from the equity calendar).
    me_date = d.groupby("ym")["date"].max()
    res["date"] = res.index.get_level_values("ym").map(me_date)

    # SET100 active-membership flag at month-end (member on the month-end trading day).
    mem = data_io.load_membership()[["date", "ric"]].drop_duplicates()
    mem_set = set(map(tuple, mem.values))
    res["is_member"] = [
        (dt, ric) in mem_set
        for ric, dt in zip(res.index.get_level_values("ric"), res["date"])
    ]
    return res, d, gm, panel, mkt_m


START_YM = pd.Period("2008-04", "M")
END_YM = pd.Period("2024-12", "M")
BETAS = ["OLS_3M_D", "OLS_1Y_D", "OLS_5Y_M"]
PREDICTORS = BETAS + ["ME", "DIVPAY", "TO", "MOM", "IVOL", "ILLIQ", "DFY"]
TARGET = "beta_realized_fwd"


FIRM = ["ME", "DIVPAY", "TO", "MOM", "IVOL", "ILLIQ"]  # non-beta firm predictors


def cleaned_panel(res):
    """Active SET100 members with complete predictors and a valid target
    (complete-case, no imputation): the descriptive-statistics sample and the
    set of stock-months at which forecasts are produced.
    """
    ym = res.index.get_level_values("ym")
    m = res[res["is_member"] & (ym >= START_YM) & (ym <= END_YM)].copy()
    keep = m[BETAS + FIRM + [TARGET]].notna().all(axis=1)
    return m[keep]


def lstm_input_panel(res):
    """All active-period stock-months with a valid target and all ten predictors,
    regardless of SET100 membership, tagged with is_member.

    The table the LSTM lookback windows are built over: origins must be members,
    but interior lookback months may be briefly outside the index.
    """
    ym = res.index.get_level_values("ym")
    m = res[(ym >= START_YM) & (ym <= END_YM)].copy()
    keep = m[PREDICTORS + [TARGET]].notna().all(axis=1)
    return m[keep]


if __name__ == "__main__":
    res, *_ = build()
    clean = cleaned_panel(res)
    print("cleaned stock-months:", len(clean),
          "| month-ends:", clean.index.get_level_values("ym").nunique(),
          "| avg stocks/month:", round(len(clean) / clean.index.get_level_values("ym").nunique(), 1))
    print(f"\n{'var':16s}{'N':>7s}{'mean':>9s}{'std':>9s}{'min':>10s}{'median':>9s}{'max':>9s}")
    for c in [TARGET] + PREDICTORS:
        x = clean[c].dropna()
        print(f"{c:16s}{len(x):7d}{x.mean():9.3f}{x.std():9.3f}{x.min():10.3f}{x.median():9.3f}{x.max():9.3f}")
