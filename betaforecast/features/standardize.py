"""Leak-free predictor standardisation for the LSTM input.

Two schemes, neither using information from beyond the forecast origin:
the nine firm/beta predictors are rank-standardised to [-1, +1] within each
month (Gu-Kelly-Xiu 2020; per-month and self-contained, so applied once on the
panel), while the macro DFY series is min-max scaled with a min/max fit on each
walk-forward training fold only (so it is applied inside the loop and test DFY
may fall slightly outside [-1, 1]).
"""
import numpy as np
import pandas as pd

from betaforecast.features import sequences as S

PREDICTORS = S.PREDICTORS                       # canonical order, DFY last
DFY = "DFY"
DFY_IDX = PREDICTORS.index(DFY)                 # 9
RANK_PREDICTORS = [p for p in PREDICTORS if p != DFY]   # the nine firm/beta preds


def rank_to_unit(x):
    """Rank-standardise a 1-D cross-section to [-1, +1] via 2*(rank-1)/(n-1) - 1.

    Ties averaged; a singleton cross-section maps to 0; missing stays missing.
    """
    x = pd.Series(x)
    n = int(x.notna().sum())
    if n <= 1:
        return pd.Series(np.where(x.notna(), 0.0, np.nan), index=x.index)
    r = x.rank(method="average")
    return 2.0 * (r - 1.0) / (n - 1.0) - 1.0


def standardize_panel(df, predictors=RANK_PREDICTORS):
    """Rank-standardise the nine firm/beta predictors per month; returns a copy.

    The ranking spans all stocks present that month (members and non-member
    lookback stocks alike). DFY is left raw -- it is min-max scaled per fold.
    """
    out = df.copy()
    for c in predictors:
        out[c] = out.groupby("ym")[c].transform(rank_to_unit)
    return out


def fit_dfy(X_train, dfy_idx=DFY_IDX):
    """Fit the DFY min-max on the training tensor only (leak-free).

    Returns ``{'min', 'max'}``; a degenerate range falls back to a unit span.
    """
    vals = X_train[:, :, dfy_idx]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    return {"min": lo, "max": hi if hi > lo else lo + 1.0}


def transform_dfy(X, params, dfy_idx=DFY_IDX):
    """Apply a fitted DFY min-max scaling to [-1, 1]; returns a copy."""
    Xz = X.copy()
    lo, hi = params["min"], params["max"]
    Xz[:, :, dfy_idx] = 2.0 * (Xz[:, :, dfy_idx] - lo) / (hi - lo) - 1.0
    return Xz


def fold_inputs(X, y, meta, iteration):
    """Model-ready train/val/test tensors for one walk-forward iteration.

    ``X`` must already be rank-standardised; the DFY min-max is fit on this
    iteration's training fold and applied to all three splits (no look-ahead).
    """
    tr, va, te = S.split_indices(meta, iteration)
    params = fit_dfy(X[tr])
    return {
        "iter": iteration["iter"],
        "params": params,
        "X_train": transform_dfy(X[tr], params), "y_train": y[tr],
        "X_val":   transform_dfy(X[va], params), "y_val":   y[va],
        "X_test":  transform_dfy(X[te], params), "y_test":  y[te],
        "meta_train": meta[tr].reset_index(drop=True),
        "meta_val":   meta[va].reset_index(drop=True),
        "meta_test":  meta[te].reset_index(drop=True),
    }


def build_model_inputs(df=None):
    """Load the panel, rank-standardise, build sequences: (X, y, meta).

    The nine firm/beta predictors come back in [-1, +1]; DFY is still raw
    (scaled per fold by ``fold_inputs``).
    """
    if df is None:
        df = S.load_panel()
    return S.build_sequences(standardize_panel(df))


def report():
    """Verify the standardised LSTM input and print the per-iteration tensors."""
    df = S.load_panel()
    X, y, meta = build_model_inputs(df)
    print(f"Standardised LSTM input: X {X.shape}, y {y.shape}\n")

    # --- Rank predictors: bounded in [-1, 1], pooled cell mean ~0 -------------
    print("Rank-standardised firm/beta predictors (over all sequence cells):")
    print(f"{'predictor':12s}{'min':>9s}{'max':>9s}{'mean':>9s}")
    for p in RANK_PREDICTORS:
        col = X[:, :, PREDICTORS.index(p)]
        print(f"{p:12s}{col.min():9.3f}{col.max():9.3f}{col.mean():9.3f}")
    dfy = X[:, :, DFY_IDX]
    print(f"\nDFY (raw, before per-fold min-max): "
          f"min {dfy.min():.4f}  max {dfy.max():.4f}  mean {dfy.mean():.4f}")

    # --- DFY min-max fit on each training fold, applied to val/test -----------
    print(f"\n{'iter':>4} {'train':>7} {'val':>6} {'test':>6} "
          f"{'DFY min':>9} {'DFY max':>9} {'train DFY*':>11} {'test DFY*':>10}")
    for it in S.walk_forward_iterations():
        f = fold_inputs(X, y, meta, it)
        p = f["params"]
        tr_z = f["X_train"][:, :, DFY_IDX]      # scaled -> [-1, 1] on train fold
        te_z = f["X_test"][:, :, DFY_IDX]       # scaled with TRAIN min/max
        print(f"{f['iter']:>4} {len(f['y_train']):>7,} {len(f['y_val']):>6,} "
              f"{len(f['y_test']):>6,} {p['min']:>9.4f} {p['max']:>9.4f} "
              f"{tr_z.mean():>11.3f} {te_z.mean():>10.3f}")
    print("\nNotes: rank predictors are in [-1, 1] with pooled mean ~0; per fold "
          "DFY is\nmin-max scaled to [-1, 1] on the TRAIN fold (col 'train DFY*' "
          "spans that\nrange), and the test fold uses the TRAIN min/max (no "
          "look-ahead), so it can\nfall outside [-1, 1] (col 'test DFY*').")


if __name__ == "__main__":
    report()
