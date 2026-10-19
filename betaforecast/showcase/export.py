"""Run every registered embed writer and write deterministic JSON."""
import json
from pathlib import Path

from betaforecast.evaluation import evaluate
from betaforecast.paths import SHOWCASE, rel
from betaforecast.showcase.embeds import EMBEDS


def dumps(payload):
    """Fixed key order (insertion), 2-space indent, trailing newline."""
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def export(out=SHOWCASE):
    """Write ``<out>/data/<embed>.json`` for every embed; return the paths."""
    data_dir = Path(out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate.load()
    written = []
    for name, writer in EMBEDS.items():
        path = data_dir / f"{name}.json"
        path.write_text(dumps(writer(df)), encoding="utf-8", newline="\n")
        written.append(path)
    return written


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m betaforecast.showcase")
    ap.add_argument("--out", default=SHOWCASE, type=Path,
                    help="showcase folder to write into (default: repo showcase/)")
    args = ap.parse_args(argv)
    for path in export(args.out):
        print(f"wrote {rel(path)}")
