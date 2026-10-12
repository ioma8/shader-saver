"""Extract the FiveK parquet shards into plain image pairs.

The HuggingFace dataset stores each example as two encoded images (`img`, the
flat input, and `gt`, the retouched target). Reading parquet inside a training
loop is slow and awkward, so this writes them once as JPEGs:

    datasets/fivek/pairs/00000_in.jpg
    datasets/fivek/pairs/00000_gt.jpg
    datasets/fivek/pairs/index.txt      # one "NNNNN" id per line

Run:

    .venv-train/bin/python tools/fivek/prepare.py
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets" / "fivek"
LONG_SIDE = 512


def _decode(value: object) -> Image.Image:
    payload = value["bytes"] if isinstance(value, dict) else value
    image = Image.open(io.BytesIO(payload))
    return image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=None, help="defaults to <source>/pairs")
    args = parser.parse_args()
    out = args.out or (args.source / "pairs")
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted((args.source / "data").glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.source / 'data'}")

    index: list[str] = []
    counter = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["img", "gt"])
        images = table.column("img").to_pylist()
        targets = table.column("gt").to_pylist()
        for raw_input, raw_target in zip(images, targets):
            identifier = f"{counter:05d}"
            for suffix, raw in (("in", raw_input), ("gt", raw_target)):
                image = _decode(raw)
                image.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
                image.save(out / f"{identifier}_{suffix}.jpg", quality=95)
            index.append(identifier)
            counter += 1
        print(f"{shard.name}: {counter} pairs so far", flush=True)

    (out / "index.txt").write_text("\n".join(index) + "\n")
    print(f"wrote {counter} pairs to {out}")


if __name__ == "__main__":
    main()
