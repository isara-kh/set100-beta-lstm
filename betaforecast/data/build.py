"""End-to-end data build: saves the two processed panels + descriptive tables."""
import os

import numpy as np
import pandas as pd

from betaforecast.data import sources as data_io, returns as R, panel as B
from betaforecast.paths import DATA_PROCESSED as OUT
START, END = "2008-04-01", "2024-12-31"


def desc(x):
    x = pd.Series(x).dropna()
    return [len(x), x.mean(), x.std(), x.min(), x.median(), x.max()]


def returns_table():
    stock_daily, mkt_excess, rf, raw_mkt = R.build_daily_returns()
    mem = data_io.load_membership()
    sd = stock_daily.merge(mem[["date", "ric"]], on=["date", "ric"], how="inner")
    sd = sd[(sd["date"] >= START) & (sd["date"] <= END)].copy()
    sd["ym"] = sd["date"].dt.to_period("M")
    sd["r_simple"] = np.exp(sd["r"]) - 1.0
    mkt_p = raw_mkt[(raw_mkt.index >= START) & (raw_mkt.index <= END)]
    rf_p = rf.reindex(mkt_p.index).dropna()  # rf on equity trading days

    # Stock series = log return of the EQUAL-WEIGHTED portfolio of the estimation
    # sample (the cleaned-panel stock-months that actually fit the betas and the
    # LSTM). The portfolio is formed in simple returns each day and then
    # log-differenced, so the volatility drag is set by the diversified portfolio
    # variance rather than by single-stock variance. Monthly = sum of daily logs.
    panel = pd.read_parquet(os.path.join(OUT, "processed_data_panel.parquet"))
    keys = set(zip(panel["ric"], pd.PeriodIndex(panel["ym"], freq="M")))
    in_panel = [(r, y) in keys for r, y in zip(sd["ric"], sd["ym"])]
    ew_d = np.log1p(sd[in_panel].groupby("date")["r_simple"].mean())
    ew_m = ew_d.groupby(ew_d.index.to_period("M")).sum()

    rows = {
        ("A", "Stocks in SET100"): desc(ew_d),
        ("A", "SET100 Index"): desc(mkt_p),
        ("A", "Risk-free rate"): desc(rf_p),
        ("B", "Stocks in SET100"): desc(ew_m),
        ("B", "SET100 Index"): desc(mkt_p.groupby(mkt_p.index.to_period("M")).sum()),
        ("B", "Risk-free rate"): desc(rf_p.groupby(rf_p.index.to_period("M")).sum()),
    }
    return pd.DataFrame(rows, index=["N", "Mean", "Std", "Min", "Median", "Max"]).T


def main():
    res, *_ = B.build()
    clean = B.cleaned_panel(res)

    # ---- Save processed panel (complete-case SET100 member sample) ----
    keep_cols = B.PREDICTORS + [B.TARGET, "date", "n_daily_month"]
    panel_out = clean[keep_cols].reset_index()
    panel_out["ym"] = panel_out["ym"].astype(str)
    panel_out.to_parquet(os.path.join(OUT, "processed_data_panel.parquet"))

    # ---- Save LSTM-input panel (target + 10 predictors, any membership) ----
    tp = B.lstm_input_panel(res)
    tp_out = tp[keep_cols + ["is_member"]].reset_index()
    tp_out["ym"] = tp_out["ym"].astype(str)
    tp_out.to_parquet(os.path.join(OUT, "lstm_input_panel.parquet"))

    # ---- Descriptive tables ----
    rt = returns_table()
    print("\n===== Table: Descriptive Statistics of Daily and Monthly Returns =====")
    for (panel, name), r in rt.iterrows():
        pct = lambda v: f"{v*100:.3f}%"
        print(f"[{panel}] {name:15s} N={int(r['N']):>7,} Mean={pct(r['Mean']):>9} "
              f"Std={pct(r['Std']):>8} Min={pct(r['Min']):>9} Med={pct(r['Median']):>8} Max={pct(r['Max']):>8}")

    print("\n===== Table: Descriptive Statistics of Target and Predictors =====")
    print(f"{'var':16s}{'N':>7s}{'mean':>10s}{'std':>10s}{'min':>11s}{'median':>10s}{'max':>10s}")
    order = [B.TARGET] + B.PREDICTORS
    for c in order:
        d = desc(clean[c])
        print(f"{c:16s}{int(d[0]):7d}{d[1]:10.4f}{d[2]:10.4f}{d[3]:11.4f}{d[4]:10.4f}{d[5]:10.4f}")

    print(f"\nStock-months: {len(clean):,} | month-ends: {clean.index.get_level_values('ym').nunique()}"
          f" | avg stocks/month: {len(clean)/clean.index.get_level_values('ym').nunique():.1f}")
    print(f"Saved processed_data_panel.parquet  shape={panel_out.shape}")
    print(f"Saved lstm_input_panel.parquet      shape={tp_out.shape} "
          f"({int(tp_out['is_member'].sum()):,} member, {int((~tp_out['is_member']).sum()):,} non-member)")


if __name__ == "__main__":
    main()
