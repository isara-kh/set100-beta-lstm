"""Writer for the committed predictor-importance CSV.

Kept apart from zero_out.py so that importing it never pulls in TensorFlow or
Keras: the showcase export and the tests read the CSV without the models.
"""
import pandas as pd

from betaforecast.paths import PREDICTOR_IMPORTANCE_CSV

# Output columns: predictor name, VI (zero-out variable importance: VW-MSE of the
# muted ensemble minus the baseline), the Diebold-Mariano t-statistic of the
# monthly loss differential, and its two-sided p-value.
COLUMNS = ["predictor", "VI", "t_stat", "p_value"]


def write_importance_csv(imp, path=PREDICTOR_IMPORTANCE_CSV):
    """Write the importance table (index `predictor`; columns VI_zeroout,
    t_stat, p_value) as a CSV with the COLUMNS header.

    Floats use 17 significant digits (round-trips a float64) and the line
    terminator is fixed, so reruns on the same numbers are byte-identical.
    """
    out = imp.rename(columns={"VI_zeroout": "VI"}).reset_index()[COLUMNS]
    out.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    return out
