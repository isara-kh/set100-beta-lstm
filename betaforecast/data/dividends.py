"""Reconstruct cash-dividend yields per (ric, trading day) from the SETSMART
Dividends.xlsx export, to turn split-adjusted price returns into total returns.

Mapping: RIC base ticker (strip '.BK' and any '^XYY' delisted suffix) == SETSMART
Symbol, with one validated rename (PTTC -> PTTCH). Per-event dividend return uses
the split-invariant ratio DPS / Price-Before-X-Date.
"""
import os

import numpy as np
import pandas as pd

from betaforecast.paths import DATA_RAW as RAW

MAX_YIELD = 0.50          # cap per-event yield (neutralises PBX data errors)
RENAMES = {"PTTC": "PTTCH"}  # RIC base ticker -> SETSMART dividend symbol


def ric_base(ric):
    return ric.split(".BK")[0]


def _load_cash_dividends():
    d = pd.read_excel(os.path.join(RAW, "Dividends.xlsx"))
    d["X-Date"] = pd.to_datetime(d["X-Date"], errors="coerce").dt.normalize()
    d["DPS"] = pd.to_numeric(d["Dividend (per Share)"], errors="coerce")
    d["PBX"] = pd.to_numeric(d["Price Before X-Date"], errors="coerce")
    d["Symbol"] = d["Symbol"].astype(str)
    cash = d[(d["Unit"] == "Baht") & d["X-Date"].notna() & d["DPS"].notna()].copy()
    return cash


def dividend_yield_daily(price_dates_by_ric, close_by_ric=None, verbose=True):
    """Return long DataFrame [ric, date, div_yield] aligned to trading days.

    price_dates_by_ric : dict ric -> sorted DatetimeIndex of that ric's trading days.
    close_by_ric       : dict ric -> close Series (for PBX-missing fallback).
    """
    cash = _load_cash_dividends()
    by_sym = {s: g for s, g in cash.groupby("Symbol")}

    rows = []
    capped, dropped_pbx, n_events = [], 0, 0
    for ric, dates in price_dates_by_ric.items():
        sym = RENAMES.get(ric_base(ric), ric_base(ric))
        if sym not in by_sym:
            continue
        dv = by_sym[sym]
        dv = dv[(dv["X-Date"] >= dates.min()) & (dv["X-Date"] <= dates.max())]
        for _, r in dv.iterrows():
            pbx = r["PBX"]
            if pd.notna(pbx) and pbx > 0:
                y = r["DPS"] / pbx
            elif close_by_ric is not None:  # fallback: DPS / our prior close
                s = close_by_ric[ric]
                prior = s[s.index < r["X-Date"]]
                if len(prior) == 0 or prior.iloc[-1] <= 0:
                    dropped_pbx += 1
                    continue
                y = r["DPS"] / prior.iloc[-1]
            else:
                dropped_pbx += 1
                continue
            if y > MAX_YIELD:
                capped.append((ric, r["X-Date"].date(), round(y, 3)))
                y = MAX_YIELD
            # align ex-date to first trading day >= X-Date
            pos = dates.searchsorted(r["X-Date"])
            if pos >= len(dates):
                continue
            rows.append((ric, dates[pos], y))
            n_events += 1

    out = (pd.DataFrame(rows, columns=["ric", "date", "div_yield"])
           .groupby(["ric", "date"], as_index=False)["div_yield"].sum())
    if verbose:
        print(f"[dividends] {n_events} cash-dividend events applied to "
              f"{out['ric'].nunique()} stocks; capped {len(capped)} at {MAX_YIELD:.0%}; "
              f"dropped {dropped_pbx} (no usable price).")
        if capped:
            print("  capped (in-universe) events:", capped)
    return out
