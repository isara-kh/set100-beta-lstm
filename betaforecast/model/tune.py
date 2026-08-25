"""LSTM hyperparameter grid search (offline experiment; the locked model in
lstm.py is untouched).

Sweeps HYPERPARAM_GRID over the full walk-forward, recording one row per
(combo, step) -- validation and test VW-MSE plus value-weight sums -- so the
results can be pooled under either selection scheme: "global-fixed" (one combo
for every step, as production does) or "per-refit" (re-pick per step by that
step's validation score). Selection is always by validation VW-MSE, so there is
no test-set leakage. Results -> lstm_tuning_results.csv, rewritten after each
combo, so an interrupted sweep resumes by skipping completed combos.
"""
import argparse
import itertools
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # quiet TF info logs

import numpy as np
import pandas as pd
import keras
from keras import layers

from betaforecast.features import sequences as S
from betaforecast.features import standardize as Z
from betaforecast.model.lstm import value_weights, train_model
from betaforecast.paths import TUNING_RESULTS as OUT, REPO
L = S.L
N_FEAT = len(S.PREDICTORS)

HYPERPARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1],
    "lr": [1e-4, 1e-3],
    "dropout": [0.0],
    "batch_size": [64, 128, 256],
}
KEYS = list(HYPERPARAM_GRID)


def combo_key(d):
    """Native-typed combo key, so rows reloaded from CSV (numpy types) still
    match combos built in memory and resumed runs skip correctly.
    """
    return (int(d["hidden_size"]), int(d["num_layers"]), float(d["lr"]),
            float(d["dropout"]), int(d["batch_size"]))


def build_model(seed, hidden_size, num_layers, lr, dropout):
    """Build an LSTM with `num_layers` stacked recurrent layers; every layer but
    the last returns full sequences. `seed` fixes the weight initialisation.
    """
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(L, N_FEAT))
    x = inp
    for i in range(num_layers):
        x = layers.LSTM(hidden_size, dropout=dropout,
                        return_sequences=(i < num_layers - 1))(x)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr), loss="mse")
    return model


def eval_combo(hp, X, y, meta, iters, n_seeds, max_epochs):
    """Run the full walk-forward for one combo: one record per step with the
    ensemble's val/test VW-MSE and value-weight sums (for exact pooling later).
    """
    recs = []
    for it in iters:
        f = Z.fold_inputs(X, y, meta, it)
        sw_tr = value_weights(f["meta_train"])
        w_va = value_weights(f["meta_val"])
        w_te = value_weights(f["meta_test"])
        v_preds, t_preds = [], []
        for seed in range(n_seeds):
            model = build_model(seed, hp["hidden_size"], hp["num_layers"],
                                hp["lr"], hp["dropout"])
            train_model(model, f, max_epochs=max_epochs, batch=hp["batch_size"],
                        verbose=0, sw_train=sw_tr, sw_val=w_va)
            v_preds.append(model.predict(f["X_val"], verbose=0).ravel())
            t_preds.append(model.predict(f["X_test"], verbose=0).ravel())
        vhat = np.mean(v_preds, axis=0)                     # ensemble = seed mean
        that = np.mean(t_preds, axis=0)
        recs.append({
            "iter": int(f["iter"]),
            "val_vwmse": float(np.average((f["y_val"] - vhat) ** 2, weights=w_va)),
            "val_wsum": float(w_va.sum()),
            "test_vwmse": float(np.average((f["y_test"] - that) ** 2, weights=w_te)),
            "test_wsum": float(w_te.sum()),
        })
    return recs


def per_refit_summary(res, iters_done):
    """Print each step's validation winner and their pooled OOS test VW-MSE."""
    print("\nPer-refit selection (design B) -- winner per walk-forward iteration:")
    num = den = 0.0
    for it in sorted(iters_done):
        sub = res[res["iter"] == it]
        w = sub.loc[sub["val_vwmse"].idxmin()]
        num += w["test_vwmse"] * w["test_wsum"]
        den += w["test_wsum"]
        print(f"  iter {int(it)}: hidden={int(w.hidden_size)} layers={int(w.num_layers)} "
              f"lr={w.lr} drop={w.dropout} batch={int(w.batch_size)}  "
              f"val={w.val_vwmse:.5f} test={w.test_vwmse:.5f}")
    print(f"Per-refit pooled OOS test VW-MSE = {num / den:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="quick check: 2 combos, 1 iter, 1 seed, 5 epochs, no save")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=1000)
    args = ap.parse_args()

    combos = [dict(zip(KEYS, vals))
              for vals in itertools.product(*HYPERPARAM_GRID.values())]
    iters = S.walk_forward_iterations()
    n_seeds, max_epochs = args.seeds, args.max_epochs
    if args.smoke:
        combos, iters, n_seeds, max_epochs = combos[:2], iters[:1], 1, 5

    rows, done = [], set()
    if os.path.exists(OUT) and not args.smoke:
        prev = pd.read_csv(OUT)
        for _, grp in prev.groupby(KEYS):
            if len(grp) == len(iters):                       # complete combo only
                done.add(combo_key(grp.iloc[0]))
                rows.extend(grp.to_dict("records"))

    total_fits = len(combos) * len(iters) * n_seeds
    print(f"{len(combos)} combos x {len(iters)} iters x {n_seeds} seeds "
          f"= {total_fits:,} fits (≤{max_epochs} epochs each); per-refit design B\n",
          flush=True)

    X, y, meta = Z.build_model_inputs()
    for i, hp in enumerate(combos, 1):
        if combo_key(hp) in done:
            print(f"[{i}/{len(combos)}] skip (already done) {hp}", flush=True)
            continue
        t0 = time.time()
        recs = eval_combo(hp, X, y, meta, iters, n_seeds, max_epochs)
        dt = round(time.time() - t0, 1)
        for r in recs:
            rows.append({**hp, **r, "seconds": dt})
        if not args.smoke:
            pd.DataFrame(rows).to_csv(OUT, index=False)
        pooled_val = (sum(r["val_vwmse"] * r["val_wsum"] for r in recs)
                      / sum(r["val_wsum"] for r in recs))
        print(f"[{i}/{len(combos)}] {hp}  pooled_val_vwmse={pooled_val:.5f}  "
              f"[{dt:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    per_refit_summary(res, res["iter"].unique())
    if not args.smoke:
        print(f"\nResults -> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
