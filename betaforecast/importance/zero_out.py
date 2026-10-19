"""Zero-out predictor importance for the trained LSTM ensemble.

For each predictor, replace its value across the lookback window with its
training-fold mean (not zero: the training sample is large-cap-biased, so e.g.
ME averages ~+0.49 after rank-standardisation), re-run the 100 saved models, and
score VI = VW-MSE(muted) - VW-MSE(baseline); a larger gap = a more important
predictor. Reuses models/ without retraining (~5 min on CPU); run after
betaforecast.model.lstm and betaforecast.benchmarks.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl"))

import numpy as np
import pandas as pd
import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from betaforecast.features import sequences as S
from betaforecast.features import standardize as Z
from betaforecast.evaluation import dm_test as DM
from betaforecast.importance.table import write_importance_csv
from betaforecast.paths import (MODELS as MODEL_DIR, OOS_FORECASTS as OOS,
                                FIGURES, PREDICTOR_IMPORTANCE as TABLE, PREDICTOR_IMPORTANCE_CSV as TABLE_CSV,
                                REPO)

FIG = FIGURES / "predictor_importance.pdf"

PREDICTORS = Z.PREDICTORS                 # the 10 predictors, canonical order
N_SEEDS = 10
PRED_BATCH = 2048                         # large batch -> faster CPU inference


def load_ensemble(k):
    """The ten seed models for walk-forward iteration k."""
    return [keras.models.load_model(os.path.join(MODEL_DIR, f"lstm_iter{k}_seed{s}.keras"))
            for s in range(N_SEEDS)]


def ensemble_predict(models, X):
    """Ensemble forecast = simple mean of the ten seed forecasts."""
    return np.mean([m.predict(X, verbose=0, batch_size=PRED_BATCH).ravel()
                    for m in models], axis=0)


def collect_forecasts():
    """Run baseline + every muted condition through the saved ensembles.

    Returns a (ric, ym)-keyed DataFrame with one yhat column per condition:
    'base' and 'zero_<pred>'. Each test sample is forecast by its own iteration's
    ensemble, exactly as in betaforecast.model.lstm.run_walkforward.
    """
    X, y, meta = Z.build_model_inputs()
    frames = []
    for it in S.walk_forward_iterations():
        k = it["iter"]
        f = Z.fold_inputs(X, y, meta, it)
        Xte, mt = f["X_test"], f["meta_test"]
        models = load_ensemble(k)

        cols = {"ric": mt["ric"].to_numpy(), "ym": mt["ym"].astype(str).to_numpy(),
                "base": ensemble_predict(models, Xte)}
        for j, p in enumerate(PREDICTORS):
            muted = Xte.copy()
            muted[:, :, j] = f["X_train"][:, :, j].mean()      # empirical train-fold mean
            cols[f"zero_{p}"] = ensemble_predict(models, muted)

        frames.append(pd.DataFrame(cols))
        print(f"iter {k}: {len(mt):,} test samples scored", flush=True)
        del models
        keras.backend.clear_session()
    return pd.concat(frames, ignore_index=True)


def vw_mse_from_se(ym, w, se):
    """Value-weighted MSE = simple mean over months of sum_i w_i * se_i."""
    monthly = pd.DataFrame({"ym": ym, "vw": w * se}).groupby("ym")["vw"].sum()
    return monthly.mean()


def importance_table(fc):
    """Per-predictor VI (VW-MSE gap vs baseline) with a DM-style t-stat.

    The monthly muted-minus-baseline loss differential is exactly the object the
    Diebold-Mariano test operates on (the network is fixed, only the forecasts
    differ), so significance is the DM statistic of that series against zero.
    Returns the table (VI, t-stat, p-value) and the baseline VW-MSE.
    """
    oos = pd.read_parquet(OOS)[["ric", "ym", "ME", "y_true", "yhat_lstm"]]
    df = fc.merge(oos, on=["ric", "ym"], how="left", validate="one_to_one")

    # baseline must reproduce the saved LSTM forecasts exactly
    diff = np.abs(df["base"] - df["yhat_lstm"]).max()
    assert diff < 1e-5, f"baseline does not reproduce saved forecasts (max |diff|={diff:.2e})"

    df["w"] = df.groupby("ym")["ME"].transform(lambda s: np.exp(s) / np.exp(s).sum())
    base_se = (df["y_true"] - df["base"]) ** 2
    base = vw_mse_from_se(df["ym"], df["w"], base_se)

    rows = []
    for p in PREDICTORS:
        zero_se = (df["y_true"] - df[f"zero_{p}"]) ** 2
        vi = vw_mse_from_se(df["ym"], df["w"], zero_se) - base
        # monthly value-weighted loss differential (muted j - baseline), DM-tested vs 0
        d_month = (df["w"] * (zero_se - base_se)).groupby(df["ym"]).sum().sort_index()
        dbar, dm, p_val, _ = DM.dm_stat(d_month)
        assert abs(dbar - vi) < 1e-9, f"{p}: monthly mean {dbar} != VI {vi}"
        rows.append({"predictor": p, "VI_zeroout": vi, "t_stat": dm, "p_value": p_val})
    out = pd.DataFrame(rows).set_index("predictor")
    return out, base


def plot_importance(imp):
    """Ranked horizontal bar chart of the value-weighted zero-out importance."""
    s = imp["VI_zeroout"].sort_values(ascending=True)
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in s.values]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta$ VW-MSE vs baseline")
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def report():
    fc = collect_forecasts()
    imp, base = importance_table(fc)
    imp.to_parquet(TABLE)
    write_importance_csv(imp, TABLE_CSV)

    print(f"\nBaseline VW-MSE = {base:.4f}  (reproduces saved LSTM forecasts)\n")
    ranked = imp.sort_values("VI_zeroout", ascending=False)
    print(f"{'predictor':12s}{'VI_zeroout':>12s}{'t-stat':>9s}{'p':>8s}")
    for p_, r in ranked.iterrows():
        star = ("***" if r.p_value < 0.01 else "**" if r.p_value < 0.05
                else "*" if r.p_value < 0.10 else "")
        print(f"{p_:12s}{r['VI_zeroout']:>12.4f}{r['t_stat']:>9.2f}{r['p_value']:>8.3f} {star}")

    plot_importance(imp)
    print(f"\nFigure -> {os.path.relpath(FIG, REPO)}   Table -> {os.path.relpath(TABLE, REPO)}, {os.path.relpath(TABLE_CSV, REPO)}")


if __name__ == "__main__":
    report()
