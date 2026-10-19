"""Showcase export: precomputed JSON for the portfolio site's Live Embeds.

Run as ``uv run python -m betaforecast.showcase [--out DIR]``. Reads only the
committed out-of-sample forecast table (ADR-0001); never the licensed panel,
Keras or the trained models.

Adding an embed: write a function ``name(df) -> dict`` in ``embeds.py`` (``df``
is the evaluation module's loaded forecast table) and decorate it with
``@embed("name")``. The export writes each registered embed to
``<out>/data/<name>.json``; the LSTM schematic is written to
``<out>/figures/lstm-ensemble.svg``.
"""
from betaforecast.showcase.export import export  # noqa: F401
