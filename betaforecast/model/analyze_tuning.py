"""Summarise lstm_tuning_results.csv: pooled OOS test VW-MSE under global-fixed
selection (design A), per-refit selection (design B), and the locked production
baseline read back from its saved forecasts. Read-only; writes nothing.
"""
import os

import numpy as np
import pandas as pd

from betaforecast.model.lstm import value_weights, OOS_OUT
from betaforecast.paths import TUNING_RESULTS as CSV
KEYS = ["hidden_size", "num_layers", "lr", "dropout", "batch_size"]


def wmean(vwmse, wsum):
    return float((vwmse * wsum).sum() / wsum.sum())


df = pd.read_csv(CSV)

# --- per-combo pooled val/test (design A candidates) ---
rows = []
for key, g in df.groupby(KEYS):
    rows.append((*key, wmean(g.val_vwmse, g.val_wsum),
                 wmean(g.test_vwmse, g.test_wsum)))
agg = pd.DataFrame(rows, columns=KEYS + ["pooled_val", "pooled_test"]).sort_values("pooled_val")

print("=== Full ranking by pooled validation VW-MSE (design A view) ===")
show = agg.copy()
show["lr"] = show["lr"].map(lambda x: f"{x:.0e}")
print(show[["hidden_size", "lr", "batch_size", "pooled_val", "pooled_test"]]
      .to_string(index=False, float_format=lambda x: f"{x:.5f}"))

a = agg.iloc[0]
print(f"\nDesign A (global-fixed) winner = best pooled validation:")
print(f"  hidden={int(a.hidden_size)} lr={a.lr:.0e} batch={int(a.batch_size)}  "
      f"pooled_val={a.pooled_val:.5f}  pooled_test={a.pooled_test:.5f}")

# --- design B: per-iteration winner by validation, pool its test ---
num = den = 0.0
for it, g in df.groupby("iter"):
    w = g.loc[g.val_vwmse.idxmin()]
    num += w.test_vwmse * w.test_wsum
    den += w.test_wsum
b_test = num / den
print(f"\nDesign B (per-refit) pooled OOS test VW-MSE = {b_test:.5f}")

# --- baseline: locked production model OOS forecasts ---
oos = pd.read_parquet(OOS_OUT)
num = den = 0.0
for it, g in oos.groupby("iter"):
    w = value_weights(g[["ric", "ym"]])
    err = (g["y_true"].to_numpy() - g["yhat_lstm"].to_numpy()) ** 2
    num += float((w * err).sum())
    den += float(w.sum())
base_test = num / den
print(f"Locked baseline ({os.path.basename(OOS_OUT)}) pooled OOS test VW-MSE = {base_test:.5f}")

print("\n=== Bottom line (pooled value-weighted OOS test MSE; lower = better) ===")
for name, v in [("Locked production baseline", base_test),
                ("Design A best (global-fixed)", a.pooled_test),
                ("Design B (per-refit)", b_test)]:
    print(f"  {name:32s} {v:.5f}")
