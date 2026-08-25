"""LSTM inputs: strict 12-month lookback sequences + walk-forward splits.

Turns the flat stock-month panel into one L x 10 matrix per forecast origin
(L = 12 months of the ten predictors) paired with the next-12-month realized
beta, and defines the ten walk-forward train/val/test splits. An origin (i, t)
is kept only if t is a SET100 member-month and the 12 months ending at t are
present and gap-free; interior lookback months may be non-member.
"""
import numpy as np
import pandas as pd

from betaforecast.paths import LSTM_INPUT_PANEL as PANEL

PREDICTORS = ["OLS_3M_D", "OLS_1Y_D", "OLS_5Y_M", "ME", "DIVPAY",
              "TO", "MOM", "IVOL", "ILLIQ", "DFY"]
TARGET = "beta_realized_fwd"
L = 12  # lookback length in months


def load_panel(path=PANEL):
    """Load the LSTM-input panel and parse ym as a monthly Period."""
    df = pd.read_parquet(path)
    df["ym"] = pd.PeriodIndex(df["ym"].astype(str), freq="M")
    return df.sort_values(["ric", "ym"]).reset_index(drop=True)


def build_sequences(df, L=L):
    """Strict-validity L x 10 sequences for every eligible forecast origin.

    Returns:
      X    : (n, L, 10) predictors over the lookback window
      y    : (n,)       realized beta over (t, t+12]
      meta : DataFrame  ['ric', 'ym', 'date'] aligned row-for-row with X, y
    """
    Xs, ys, rics, yms, dates = [], [], [], [], []
    for ric, g in df.groupby("ric", sort=False):
        if len(g) < L:
            continue
        P = g[PREDICTORS].to_numpy(dtype=float)
        T = g[TARGET].to_numpy(dtype=float)
        memb = g["is_member"].to_numpy(dtype=bool)
        ords = np.array([p.ordinal for p in g["ym"]])  # months since epoch
        periods = g["ym"].tolist()
        date = g["date"].tolist()
        for j in range(L - 1, len(g)):
            if not memb[j]:                              # origin must be a member-month
                continue
            # sorted, unique months => span of L-1 implies L consecutive months
            if ords[j] - ords[j - L + 1] == L - 1:
                Xs.append(P[j - L + 1: j + 1])
                ys.append(T[j])
                rics.append(ric)
                yms.append(periods[j])
                dates.append(date[j])
    X = np.asarray(Xs, dtype=float).reshape(-1, L, len(PREDICTORS))
    y = np.asarray(ys, dtype=float)
    meta = pd.DataFrame({"ric": rics, "ym": yms, "date": dates})
    return X, y, meta


def walk_forward_iterations(n_iters=10, train_years=5, first_train_start=2008,
                            expanding=False):
    """The ten walk-forward iterations, split by calendar year of the origin t.

    Iteration k: train [2008+k .. 2012+k], val 2013+k, gap 2014+k (discarded),
    test 2015+k. The gap year equals the 12-month horizon (Drobetz et al. 2025):
    it keeps the validation labels' realization window from overlapping the test
    window, so early stopping carries no look-ahead. ``expanding=True`` anchors
    the training start at 2008 instead of rolling the 5-year block; it tested
    worse (VW-MSE 0.1124 vs 0.1094), so rolling stays primary.
    """
    its = []
    for k in range(n_iters):
        train_end = first_train_start + k + train_years - 1   # last train year
        train_start = first_train_start if expanding else first_train_start + k
        its.append({
            "iter": k,
            "expanding": expanding,
            "train_years": list(range(train_start, train_end + 1)),
            "val_year": train_end + 1,
            "gap_year": train_end + 2,
            "test_year": train_end + 3,
        })
    return its


def split_indices(meta, iteration):
    """Boolean masks (train, val, test) into X/y/meta for one iteration."""
    yr = meta["ym"].dt.year.to_numpy()
    train = np.isin(yr, iteration["train_years"])
    val = yr == iteration["val_year"]
    test = yr == iteration["test_year"]
    return train, val, test


def report(df=None):
    """Print the strict-validity sample counts and per-iteration split sizes."""
    if df is None:
        df = load_panel()
    X, y, meta = build_sequences(df)
    print(f"LSTM-input months (target+10 preds) : {len(df):,}  "
          f"({int(df['is_member'].sum()):,} member, {int((~df['is_member']).sum()):,} non-member)")
    print(f"Strict L={L} sequences (member origin): {len(meta):,}  "
          f"(X shape {X.shape}, y shape {y.shape})")
    print(f"Sequence ym range                   : {meta['ym'].min()} -> {meta['ym'].max()}")

    print(f"\n{'iter':>4} {'train yrs':>13} {'train':>7} {'val':>6} {'test':>6}")
    train_counts, test_counts = [], []
    for it in walk_forward_iterations():
        tr, va, te = split_indices(meta, it)
        train_counts.append(int(tr.sum()))
        test_counts.append(int(te.sum()))
        yrs = f"{it['train_years'][0]}-{it['train_years'][-1]}"
        print(f"{it['iter']:>4} {yrs:>13} {int(tr.sum()):>7,} "
              f"{int(va.sum()):>6,} {int(te.sum()):>6,}")

    avg_train = sum(train_counts) / len(train_counts)
    total_oos = sum(test_counts)
    print(f"\nAvg training samples / iteration : {avg_train:,.0f}")
    print(f"Total OOS test forecasts 2015-2024: {total_oos:,} "
          f"({meta['ym'].dt.year.between(2015, 2024).sum():,} via year filter)")


if __name__ == "__main__":
    report()
