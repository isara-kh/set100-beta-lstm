"""Hand-written half of the Content Contract: showcase.yaml and case-study.mdx.

The prose is the owner's; these tests only pin what can silently drift: the
metadata keys, the embed ids, the MDX subset the Portfolio Site accepts, and
the numbers on the hook cards (checked against the committed export).
"""
import json
import re

import pytest
import yaml

from betaforecast.paths import SHOWCASE

YAML = SHOWCASE / "showcase.yaml"
MDX = SHOWCASE / "case-study.mdx"

EXPECTED_YAML = {
    "contract": 1,
    "slug": "set100-beta-lstm",
    "title": "Forecasting Thai stock betas with an LSTM",
    "thesis": "Deep learning beats rolling OLS beta only where OLS is noisy.",
    "market": "Thailand · SET100",
    "method": "LSTM ensemble vs rolling OLS",
    "years": {"from": 2008, "to": 2024},
    "status": "published",
    "headline": "~2× more accurate than the 5-year monthly beta",
    "links": {
        "repo": "https://github.com/isara-kh/set100-beta-lstm",
        "paper": "attachments/manuscript-redacted.pdf",
    },
    "embeds": ["walk-forward", "horse-race", "where-it-wins", "shrinkage",
               "importance"],
}

TAG = re.compile(r'<(Embed|Stat)\s+([^<>{}]*?)\s*/>')
ATTR = re.compile(r'(\w+)="([^"{}<>]*)"')
EMBED_ID = re.compile(r'<Embed\s+id="([^"]*)"\s*/>')
STAT = re.compile(r'<Stat\s+value="([^"]*)"\s+caption="([^"]*)"\s*/>')


@pytest.fixture(scope="module")
def mdx():
    return MDX.read_text(encoding="utf-8")


def export_json(name):
    return json.loads((SHOWCASE / "data" / f"{name}.json").read_text())


def cell(weighting, regime, group):
    cells = export_json("where-it-wins")["cells"]
    (hit,) = [c for c in cells if (c["weighting"], c["regime"], c["group"])
              == (weighting, regime, group)]
    return hit


def prose_without_tags_and_maths(text):
    """What is left once allowed tags, maths and escaped characters are gone."""
    text = TAG.sub("", text)
    text = re.sub(r"\$\$.+?\$\$", "", text, flags=re.S)
    text = re.sub(r"\$[^$\n]+\$", "", text)
    return re.sub(r"\\[{}<]", "", text)


# --- showcase.yaml ----------------------------------------------------------

def test_yaml_has_exactly_the_specs_keys_and_values():
    data = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    assert data == EXPECTED_YAML
    # key order mirrors the spec's yaml
    assert list(data) == list(EXPECTED_YAML)


# --- embeds -----------------------------------------------------------------

def test_mdx_embed_ids_equal_yaml_embeds_exactly_once_each(mdx):
    ids = EMBED_ID.findall(mdx)
    assert sorted(ids) == sorted(EXPECTED_YAML["embeds"])
    assert len(ids) == len(set(ids))


def test_embeds_appear_in_page_order(mdx):
    assert EMBED_ID.findall(mdx) == EXPECTED_YAML["embeds"]


def test_figure_is_a_relative_markdown_image(mdx):
    assert re.search(r"!\[[^\]]+\]\(\./figures/lstm-ensemble\.svg\)", mdx)


# --- MDX lint ---------------------------------------------------------------

def test_no_import_or_export_statements(mdx):
    assert not re.search(r"^\s*(import|export)\b", mdx, flags=re.M)


def test_only_embed_and_stat_tags(mdx):
    names = set(re.findall(r"</?([A-Za-z][\w.]*)", prose_outside_maths(mdx)))
    assert names <= {"Embed", "Stat"}
    for _, attrs in TAG.findall(mdx):
        assert set(dict(ATTR.findall(attrs))) <= {"id", "value", "caption"}


def prose_outside_maths(text):
    text = re.sub(r"\$\$.+?\$\$", "", text, flags=re.S)
    return re.sub(r"\$[^$\n]+\$", "", text)


def test_no_unescaped_braces_or_angle_brackets_outside_maths(mdx):
    rest = prose_without_tags_and_maths(mdx)
    assert not re.search(r"[{}<]", rest), re.findall(r".*[{}<].*", rest)


def test_no_raw_html_comments_or_inline_html(mdx):
    assert "<!--" not in mdx


# --- hook numbers -----------------------------------------------------------

def _ratio_vs_5y():
    c = cell("vw", "all", "all")["mse"]
    return c["ols_5y_m"] / c["lstm"]


# Every Stat value in the MDX needs an entry here: stat text -> checker that
# compares it with the committed export within the card's display rounding.
STAT_CHECKS = {
    # 0.1968 / 0.0995 = 1.98, displayed as "~2x" (nearest integer).
    "~2×": lambda: round(_ratio_vs_5y()) == 2 and abs(_ratio_vs_5y() - 2) < 0.05,
    # DM t against OLS_1Y_D, 2 dp.
    "t = 0.20": lambda: (
        f"{export_json('horse-race')['dm']['ols_1y_d']['t']:.2f}" == "0.20"
        and export_json("horse-race")["dm"]["ols_1y_d"]["p"] > 0.05),
    # LSTM beats OLS_1Y_D in small and mid caps (VW, full sample), significantly.
    "Small + mid": lambda: all(
        cell("vw", "all", g)["mse"]["lstm"] < cell("vw", "all", g)["mse"]["ols_1y_d"]
        and cell("vw", "all", g)["dm"]["ols_1y_d"]["t"] < 0
        and cell("vw", "all", g)["dm"]["ols_1y_d"]["p"] < 0.05
        for g in ("small", "mid")),
}


def test_there_are_exactly_three_hook_stats(mdx):
    assert len(STAT.findall(mdx)) == 3


def test_every_stat_has_a_checker_and_agrees_with_the_export(mdx):
    stats = STAT.findall(mdx)
    unchecked = [v for v, _ in stats if v not in STAT_CHECKS]
    assert not unchecked, f"Stat value(s) with no checker: {unchecked}"
    for value, caption in stats:
        assert caption.strip(), f"empty caption for {value}"
        assert STAT_CHECKS[value](), f"Stat {value!r} disagrees with the export"


def test_headline_in_yaml_agrees_with_the_export():
    assert round(_ratio_vs_5y()) == 2
