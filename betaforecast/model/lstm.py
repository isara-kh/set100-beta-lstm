"""The LSTM beta forecaster and its walk-forward training driver.

Forecasts one-year-ahead realized beta from 12-month sequences of ten predictors
(Gu-Kelly-Xiu 2020 / Drobetz et al. 2025 style). One small fixed architecture --
LSTM(8, dropout 0.1) -> Dense(1) -- trained with a value-weighted MSE, averaged
over a 10-seed ensemble, refit at each of the ten walk-forward steps. Forecasts
go to lstm_oos_forecasts.parquet, trained models to models/.

Full walkthrough: notebooks/02_lstm_explained.ipynb.
"""
import argparse
import os
import time

from betaforecast.paths import (REPO, MODELS as MODEL_DIR,
                                 LSTM_OOS_FORECASTS as OOS_OUT,
                                 LSTM_INPUT_PANEL as PANEL)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # quiet TF info logs
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache_mpl"))

import numpy as np
import pandas as pd
import keras
from keras import layers

from betaforecast.features import sequences as S
from betaforecast.features import standardize as Z
_RAW_ME = None  # cached (ric, ym) -> raw ME lookup for value-weighted training

# Units and dropout picked by a validation VW-MSE search over units {4,8,16} x
# dropout {0,0.1}; everything here is then held fixed for all walk-forward steps.
UNITS = 8
DROPOUT = 0.1
LR = 1e-3
BATCH = 128
MAX_EPOCHS = 1000
PATIENCE = 10
N_SEEDS = 10
L = S.L
N_FEAT = len(S.PREDICTORS)


def value_weights(meta_df):
    """Within-month market-cap share per observation (Drobetz et al. 2025).

    w_{i,t} = exp(ME_i) / sum_j exp(ME_j) over the origin month (raw ME looked up
    from the panel), then the whole vector is rescaled to mean 1 so the effective
    learning rate matches unweighted training.
    """
    global _RAW_ME
    if _RAW_ME is None:
        p = pd.read_parquet(PANEL)[["ric", "ym", "ME"]].copy()
        p["ym"] = p["ym"].astype(str)
        _RAW_ME = p.set_index(["ric", "ym"])["ME"]
    ym = meta_df["ym"].astype(str).to_numpy()
    keys = list(zip(meta_df["ric"].to_numpy(), ym))
    mc = np.exp(_RAW_ME.loc[keys].to_numpy())
    d = pd.DataFrame({"ym": ym, "mc": mc})
    w = d.groupby("ym")["mc"].transform(lambda s: s / s.sum()).to_numpy()
    return w * (len(w) / w.sum())                       # rescale so weights average 1


def build_model(seed, units=UNITS, dropout=DROPOUT, lr=LR):
    """Build one LSTM with the fixed architecture; `seed` fixes the weight init."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input(shape=(L, N_FEAT)),
        layers.LSTM(units, dropout=dropout),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr), loss="mse")
    return model


def train_model(model, fold, max_epochs=MAX_EPOCHS, batch=BATCH,
                patience=PATIENCE, verbose=0, sw_train=None, sw_val=None):
    """Fit one model with early stopping on validation loss (best weights restored).

    Optional sample weights (sw_train/sw_val) make the loss value-weighted;
    with both None it is the plain equal-weighted MSE.
    """
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True)
    val_data = ((fold["X_val"], fold["y_val"]) if sw_val is None
                else (fold["X_val"], fold["y_val"], sw_val))
    model.fit(fold["X_train"], fold["y_train"], sample_weight=sw_train,
              validation_data=val_data,
              epochs=max_epochs, batch_size=batch, callbacks=[es], verbose=verbose)
    return model


def run_walkforward(n_seeds=N_SEEDS, max_epochs=MAX_EPOCHS, iters=None,
                    save_models=True, verbose=0, out_path=OOS_OUT):
    """Train the value-weighted seed ensemble across all walk-forward steps.

    Writes one forecast row per (ric, ym) test sample to `out_path` (pass None to
    skip, e.g. for a demo run) and the trained models to models/ (reused later by
    evaluation and importance). Returns the OOS forecast table.
    """
    X, y, meta = Z.build_model_inputs()
    print(f"Model input: X {X.shape}, y {y.shape}; {n_seeds}-seed ensemble, "
          f"<= {max_epochs} epochs/fit, value-weighted loss\n")
    if save_models:
        os.makedirs(MODEL_DIR, exist_ok=True)

    all_iters = S.walk_forward_iterations()
    if iters is not None:
        all_iters = [all_iters[k] for k in iters]

    rows = []
    for it in all_iters:
        t0 = time.time()
        f = Z.fold_inputs(X, y, meta, it)
        k = it["iter"]
        sw_tr, sw_va = value_weights(f["meta_train"]), value_weights(f["meta_val"])
        seed_preds = []
        for seed in range(n_seeds):
            model = build_model(seed)
            train_model(model, f, max_epochs=max_epochs, verbose=verbose,
                        sw_train=sw_tr, sw_val=sw_va)
            seed_preds.append(model.predict(f["X_test"], verbose=0).ravel())
            if save_models:
                model.save(os.path.join(MODEL_DIR, f"lstm_iter{k}_seed{seed}.keras"))
        yhat = np.mean(seed_preds, axis=0)                  # ensemble = mean of seeds
        mt = f["meta_test"]
        mse = float(np.mean((f["y_test"] - yhat) ** 2))
        rows.append(pd.DataFrame({
            "ric": mt["ric"].to_numpy(),
            "ym": mt["ym"].astype(str).to_numpy(),
            "date": mt["date"].to_numpy(),
            "y_true": f["y_test"],
            "yhat_lstm": yhat,
            "iter": k,
        }))
        if out_path:
            pd.concat(rows, ignore_index=True).to_parquet(out_path)   # persist partial
        print(f"iter {k}: train {len(f['y_train']):,} / val {len(f['y_val']):,} "
              f"/ test {len(f['y_test']):,}  ensemble test MSE = {mse:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    oos = pd.concat(rows, ignore_index=True)
    if out_path:
        oos.to_parquet(out_path)
        print(f"\nSaved {len(oos):,} OOS forecasts -> {os.path.relpath(out_path, REPO)} "
              f"({oos['ym'].min()} -> {oos['ym'].max()})")
    return oos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="quick check: iter 0 only, 2 seeds, 5 epochs, no save")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    if args.smoke:
        run_walkforward(n_seeds=2, max_epochs=5, iters=[0],
                        save_models=False, verbose=0, out_path=None)
    else:
        run_walkforward(n_seeds=args.seeds)
