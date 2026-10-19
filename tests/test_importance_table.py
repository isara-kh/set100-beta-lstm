"""The committed importance CSV: column contract and byte-stable output.

Synthetic frame only; the zero-out step itself needs the trained models.
"""
import pandas as pd

from betaforecast.importance.table import write_importance_csv

PREDICTORS = ["OLS_3M_D", "OLS_1Y_D", "OLS_5Y_M", "ME", "DIVPAY",
              "TO", "MOM", "IVOL", "ILLIQ", "DFY"]


def synthetic():
    return pd.DataFrame(
        {"VI_zeroout": [0.1 * i + 1 / 3 for i in range(10)],
         "t_stat": [2.5 - 0.7 * i for i in range(10)],
         "p_value": [0.01 * (i + 1) for i in range(10)]},
        index=pd.Index(PREDICTORS, name="predictor"))


def test_columns_and_rows(tmp_path):
    path = tmp_path / "imp.csv"
    write_importance_csv(synthetic(), path)
    out = pd.read_csv(path)
    assert list(out.columns) == ["predictor", "VI", "t_stat", "p_value"]
    assert out["predictor"].tolist() == PREDICTORS
    assert out.loc[0, "VI"] == 1 / 3          # full precision survives the round trip


def test_rerun_is_byte_identical(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_importance_csv(synthetic(), a)
    write_importance_csv(synthetic(), b)
    assert a.read_bytes() == b.read_bytes()
