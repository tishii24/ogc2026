#!/usr/bin/env python3
"""Generate train-empty cases by inserting empty layers into train cases."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20260628
DEFAULT_PROB = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert random empty layers into problem JSON files."
    )
    parser.add_argument(
        "--input-dir", default="train", help="Input directory. default: train"
    )
    parser.add_argument(
        "--output-dir",
        default="train-empty",
        help="Output directory. default: train-empty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=DEFAULT_PROB,
        help=f"Insertion probability per orientation. default: {DEFAULT_PROB}",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dir(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def insert_empty_layers(data: dict[str, Any], rng: random.Random, prob: float) -> int:
    inserted = 0
    first_layers: list[Any] | None = None

    for block in data.get("blocks", []):
        for orientation in block.get("shape", []):
            layers = orientation.get("layers")
            if not isinstance(layers, list):
                continue
            if first_layers is None:
                first_layers = layers
            if rng.random() < prob:
                pos = rng.randrange(len(layers))
                layers[pos] = []
                inserted += 1

    if inserted == 0 and first_layers is not None:
        first_layers.insert(0, [])
        inserted += 1

    return inserted


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.prob <= 1.0:
        print("error: --prob must be between 0 and 1", file=sys.stderr)
        return 1

    root = repo_root()
    input_dir = resolve_dir(root, args.input_dir)
    output_dir = resolve_dir(root, args.output_dir)

    if not input_dir.is_dir():
        print(f"error: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        print(f"error: no json files found in {input_dir}", file=sys.stderr)
        return 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    total_inserted = 0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        inserted = insert_empty_layers(data, rng, args.prob)
        total_inserted += inserted

        output_path = output_dir / path.name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

        print(f"{path.name}: inserted={inserted}", file=sys.stderr)

    print(
        f"created {len(paths)} files in {output_dir.relative_to(root)}; total_inserted={total_inserted}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
