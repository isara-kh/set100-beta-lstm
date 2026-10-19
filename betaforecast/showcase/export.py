"""Run every registered embed writer and write deterministic JSON."""
import json
from pathlib import Path

from betaforecast.evaluation import evaluate
from betaforecast.figures import lstm_architecture
from betaforecast.paths import SHOWCASE, rel
from betaforecast.showcase.embeds import EMBEDS


def dumps(payload):
    """Fixed key order (insertion), 2-space indent, trailing newline."""
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def export(out=SHOWCASE):
    """Write ``<out>/data/<embed>.json`` for every embed and the static
    ``<out>/figures/lstm-ensemble.svg``; return the paths."""
    data_dir = Path(out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate.load()
    written = []
    for name, writer in EMBEDS.items():
        path = data_dir / f"{name}.json"
        path.write_text(dumps(writer(df)), encoding="utf-8", newline="\n")
        written.append(path)
    # Static diagram; the writer pins the Agg backend and the SVG hash salt.
    written.append(lstm_architecture.write_svg(
        Path(out) / "figures" / lstm_architecture.SVG_NAME))
    return written


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m betaforecast.showcase")
    ap.add_argument("--out", default=SHOWCASE, type=Path,
                    help="showcase folder to write into (default: repo showcase/)")
    args = ap.parse_args(argv)
    for path in export(args.out):
        print(f"wrote {rel(path)}")
