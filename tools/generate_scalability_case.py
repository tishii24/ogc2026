#!/usr/bin/env python3

import copy
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "in/train-final/prob_23.json"
    output_path = root / "in/scalability/prob_600_10.json"

    with source_path.open(encoding="utf-8") as file:
        source = json.load(file)

    bays = copy.deepcopy(source["bays"] * 2)
    blocks = []
    for _ in range(2):
        for source_block in source["blocks"]:
            block = copy.deepcopy(source_block)
            block["bay_preferences"] *= 2
            blocks.append(block)

    problem = {
        "name": "prob_600_10",
        "bays": bays,
        "blocks": blocks,
        "weights": copy.deepcopy(source["weights"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(problem, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
