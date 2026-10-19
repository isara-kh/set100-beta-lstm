"""Central filesystem paths: every module resolves its inputs and outputs
through these constants, anchored at the repo root one level above this file.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATA = REPO / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
MODELS = REPO / "models"
FIGURES = REPO / "figures"
# Figures the README embeds. Unlike FIGURES, these are versioned: the README
# depends on them, so they must exist in a fresh clone.
DOCS_IMG = REPO / "docs" / "img"

# The Content Contract folder for the portfolio site. The showcase export owns
# data/ and figures/ inside it; attachments/ and the hand-written files are not
# touched by any code.
SHOWCASE = REPO / "showcase"
SHOWCASE_DATA = SHOWCASE / "data"
SHOWCASE_FIGURES = SHOWCASE / "figures"

# Processed artifacts referenced by name across the pipeline.
LSTM_INPUT_PANEL = DATA_PROCESSED / "lstm_input_panel.parquet"
PROCESSED_PANEL = DATA_PROCESSED / "processed_data_panel.parquet"
LSTM_OOS_FORECASTS = DATA_PROCESSED / "lstm_oos_forecasts.parquet"
OOS_FORECASTS = DATA_PROCESSED / "oos_forecasts.parquet"
MONTHLY_VW_MSE = DATA_PROCESSED / "monthly_vw_mse.parquet"
PREDICTOR_IMPORTANCE = DATA_PROCESSED / "predictor_importance.parquet"
TUNING_RESULTS = DATA_PROCESSED / "lstm_tuning_results.csv"


def rel(path) -> str:
    """Path shown relative to the repo root, for tidy log messages.

    Falls back to the absolute path when the target lies outside the repo, so a
    logging helper never brings down the caller.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)
