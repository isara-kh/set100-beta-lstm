"""Export-seam tests: run the showcase export into a temp dir, assert on output."""
import json
import re

import pytest

from betaforecast.evaluation import dm_test, evaluate
from betaforecast.features import sequences
from betaforecast.showcase import export
from betaforecast.showcase.embeds import EMBEDS


@pytest.fixture(scope="module")
def out(tmp_path_factory, oos_module):
    d = tmp_path_factory.mktemp("showcase")
    export(d)
    return d


@pytest.fixture(scope="module")
def oos_module():
    from betaforecast.paths import OOS_FORECASTS
    if not OOS_FORECASTS.exists():
        pytest.skip("oos_forecasts.parquet not present")


def load(out, name):
    return json.loads((out / "data" / f"{name}.json").read_text())


def decimals(text):
    return len(text.split(".")[1]) if "." in text else 0


def test_writes_only_the_registered_embed_files(out):
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == sorted([*(f"data/{n}.json" for n in EMBEDS),
                            "figures/lstm-ensemble.svg"])
    assert set(EMBEDS) >= {"walk-forward", "horse-race"}


def test_walk_forward_matches_the_study_scheme(out):
    wf = load(out, "walk-forward")
    assert list(wf) == ["start_year", "train_years", "validation_years",
                        "gap_years", "test_years", "iterations"]
    assert list(wf.values()) == [2008, 5, 1, 1, 1, 10]
    it = sequences.walk_forward_iterations()
    assert len(it) == wf["iterations"]
    assert it[-1]["test_year"] == wf["start_year"] + wf["iterations"] - 1 + 7


def test_horse_race_shape(out):
    hr = load(out, "horse-race")
    assert set(hr) == {"covid_start", "months", "dm"}
    assert hr["covid_start"] == dm_test.COVID
    months = hr["months"]
    assert len(months) == 120
    origins = [m["origin"] for m in months]
    assert origins == sorted(origins) and len(set(origins)) == 120
    assert all(re.fullmatch(r"\d{4}-\d{2}", o) for o in origins)
    assert origins[0] == "2015-01" and origins[-1] == "2024-12"
    for m in months:
        assert set(m) == {"origin", "d_vs_ols_1y_d", "d_vs_ols_5y_m"}
        assert isinstance(m["d_vs_ols_1y_d"], float)
    assert set(hr["dm"]) == {"ols_1y_d", "ols_5y_m"}
    for v in hr["dm"].values():
        assert set(v) == {"dbar", "t", "p"}


def test_horse_race_rounding(out):
    text = (out / "data" / "horse-race.json").read_text()
    for key, dp in (("d_vs_ols_1y_d", 6), ("d_vs_ols_5y_m", 6), ("dbar", 6),
                    ("t", 2), ("p", 3)):
        for lit in re.findall(rf'"{key}": (-?[\d.]+(?:e-?\d+)?)', text):
            assert decimals(lit) <= dp, (key, lit)


def test_horse_race_dm_and_signs(out):
    hr = load(out, "horse-race")
    assert hr["dm"]["ols_5y_m"]["t"] == pytest.approx(-6.55, abs=0.01)
    assert hr["dm"]["ols_5y_m"]["p"] < 0.001
    assert hr["dm"]["ols_1y_d"]["t"] == pytest.approx(0.20, abs=0.01)


def test_differential_means_match_published_headline_gaps(out):
    hr = load(out, "horse-race")
    mean = lambda k: sum(m[k] for m in hr["months"]) / 120
    # LSTM 0.0995, OLS_1Y_D 0.0974, OLS_5Y_M 0.1968 (4 dp each => 1e-4 slack)
    assert mean("d_vs_ols_1y_d") == pytest.approx(0.0995 - 0.0974, abs=1e-4)
    assert mean("d_vs_ols_5y_m") == pytest.approx(0.0995 - 0.1968, abs=1e-4)
    assert hr["dm"]["ols_5y_m"]["dbar"] == pytest.approx(mean("d_vs_ols_5y_m"), abs=1e-6)


def test_two_runs_are_byte_identical(out, tmp_path):
    export(tmp_path)
    for name in EMBEDS:
        a = (out / "data" / f"{name}.json").read_bytes()
        assert a == (tmp_path / "data" / f"{name}.json").read_bytes()
        assert a.endswith(b"\n") and not a.endswith(b"\n\n")


def test_svg_is_byte_identical_and_has_no_run_specific_content(out, tmp_path):
    export(tmp_path)
    a = (out / "figures" / "lstm-ensemble.svg").read_bytes()
    assert a == (tmp_path / "figures" / "lstm-ensemble.svg").read_bytes()
    text = a.decode("utf-8")
    assert "<dc:date>" not in text and "<dc:date" not in text


def test_svg_parses_with_an_svg_root(out):
    from xml.etree import ElementTree as ET
    root = ET.parse(out / "figures" / "lstm-ensemble.svg").getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"


def test_svg_export_does_not_change_global_matplotlib_state(tmp_path):
    import matplotlib
    from betaforecast.figures import lstm_architecture
    before = dict(matplotlib.rcParams)
    lstm_architecture.write_svg(tmp_path / "x.svg")
    assert dict(matplotlib.rcParams) == before
    assert matplotlib.get_backend().lower() == "agg"


def test_export_leaves_existing_showcase_files_alone(tmp_path, oos_module):
    att = tmp_path / "attachments"
    att.mkdir()
    (att / "keep.pdf").write_bytes(b"x")
    export(tmp_path)
    assert (att / "keep.pdf").read_bytes() == b"x"


def test_committed_export_is_fresh(out):
    from betaforecast.paths import SHOWCASE_DATA, SHOWCASE_FIGURES
    for name in EMBEDS:
        assert (SHOWCASE_DATA / f"{name}.json").read_bytes() == \
            (out / "data" / f"{name}.json").read_bytes()
    assert (SHOWCASE_FIGURES / "lstm-ensemble.svg").read_bytes() == \
        (out / "figures" / "lstm-ensemble.svg").read_bytes()


# --- importance embed -------------------------------------------------------

def test_importance_shape_and_sort(out):
    imp = load(out, "importance")
    assert list(imp) == ["predictors"]
    rows = imp["predictors"]
    assert len(rows) == 10
    for r in rows:
        assert list(r) == ["name", "vi", "t", "p"]
        assert isinstance(r["name"], str)
        assert all(isinstance(r[k], float) for k in ("vi", "t", "p"))
    vis = [r["vi"] for r in rows]
    assert vis == sorted(vis, reverse=True)
    assert len({r["name"] for r in rows}) == 10
    assert rows[0]["name"] == "OLS_3M_D" and min(vis) < 0


def test_importance_rounding(out):
    text = (out / "data" / "importance.json").read_text()
    for key, dp in (("vi", 6), ("t", 2), ("p", 3)):
        lits = re.findall(rf'"{key}": (-?[\d.]+(?:e-?\d+)?)', text)
        assert len(lits) == 10
        for lit in lits:
            assert decimals(lit) <= dp, (key, lit)
    first = load(out, "importance")["predictors"][0]
    assert first == {"name": "OLS_3M_D", "vi": 0.014446, "t": 4.93, "p": 0.0}


def test_missing_importance_csv_fails_clearly(tmp_path, monkeypatch, oos_module):
    from betaforecast.showcase import embeds
    monkeypatch.setattr(embeds, "PREDICTOR_IMPORTANCE_CSV", tmp_path / "predictor_importance.csv")
    with pytest.raises(FileNotFoundError) as exc:
        export(tmp_path / "site")
    msg = str(exc.value)
    assert "predictor_importance.csv" in msg
    assert "python -m betaforecast.importance.zero_out" in msg


# --- shrinkage embed --------------------------------------------------------

SERIES = ("realized", "lstm", "ols_1y_d", "ols_5y_m")

# Manuscript Table 5, as printed (4 dp): n = 10,145 pooled stock-months.
# Std is the sample (ddof=1) standard deviation; Skew is the bias-corrected
# sample skewness (pandas' default), which is what separates 0.7465 from the
# population estimator's 0.7464 for the realized beta.
TABLE_5 = {
    "realized": {"mean": 0.9753, "sd": 0.3893, "skew": 0.7465, "min": -0.5451, "max": 3.2181},
    "lstm":     {"mean": 0.9427, "sd": 0.1910, "skew": -0.4345, "min": 0.3275, "max": 1.4604},
    "ols_1y_d": {"mean": 0.9771, "sd": 0.3759, "skew": 0.5282, "min": -1.4974, "max": 3.1643},
    "ols_5y_m": {"mean": 1.1241, "sd": 0.4254, "skew": 0.3493, "min": -0.2704, "max": 2.5400},
}


def test_shrinkage_shape(out):
    sh = load(out, "shrinkage")
    assert list(sh) == ["edges", "counts", "stats", "mean_series"]
    edges = sh["edges"]
    assert edges == sorted(edges) and len(set(edges)) == len(edges)
    assert list(sh["counts"]) == list(SERIES)
    for s in SERIES:
        counts = sh["counts"][s]
        assert len(counts) == len(edges) - 1
        assert all(type(c) is int and c >= 0 for c in counts)
        assert type(sh["stats"][s]["n"]) is int
        assert {"n", "mean", "sd", "skew"} <= set(sh["stats"][s])
    ms = sh["mean_series"]
    assert len(ms) == 120
    origins = [m["origin"] for m in ms]
    assert origins == sorted(origins) and len(set(origins)) == 120
    assert origins[0] == "2015-01" and origins[-1] == "2024-12"
    assert all(re.fullmatch(r"\d{4}-\d{2}", o) for o in origins)
    assert all(list(m) == ["origin", *SERIES] for m in ms)


def test_shrinkage_counts_sum_to_n_and_cover_every_observation(out):
    sh = load(out, "shrinkage")
    for s in SERIES:
        assert sh["stats"][s]["n"] == 10145
        assert sum(sh["counts"][s]) == sh["stats"][s]["n"]
        lo, hi = sh["stats"][s]["min"], sh["stats"][s]["max"]
        assert sh["edges"][0] <= lo and hi <= sh["edges"][-1]


def test_shrinkage_rounding(out):
    text = (out / "data" / "shrinkage.json").read_text()
    sh = load(out, "shrinkage")
    # Edges are exact at 2 dp; stats and mean-series values at 4 dp.
    assert all(round(e, 2) == e for e in sh["edges"])
    for key in ("mean", "sd", "skew", "min", "max", *SERIES):
        for lit in re.findall(rf'"{key}": (-?[\d.]+(?:e-?\d+)?)', text):
            assert decimals(lit) <= 4, (key, lit)


@pytest.mark.parametrize("series", SERIES)
def test_shrinkage_stats_match_manuscript_table_5(out, series):
    got = load(out, "shrinkage")["stats"][series]
    for stat, printed in TABLE_5[series].items():
        assert got[stat] == pytest.approx(printed, abs=5e-5), (series, stat)


def test_shrinkage_mean_series_averages_to_pooled_means_within_rounding(out):
    sh = load(out, "shrinkage")
    # Every month has the same cross-section size only approximately, so the
    # mean of monthly means is near (not equal to) the pooled mean.
    for s in SERIES:
        avg = sum(m[s] for m in sh["mean_series"]) / 120
        assert avg == pytest.approx(sh["stats"][s]["mean"], abs=0.02)
