from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import load_axis_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-data", default="data/AXIS_qa_test")
    p.add_argument("--train-data", default="data/anomaly_llava_training_dataset")
    p.add_argument("--output", default="experiments/reproduction/manifests")
    p.add_argument("--seed", type=int, default=72)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for subset in ("paper140", "full"):
        records = load_axis_records(a.test_data, subset=subset, seed=42)
        (out / f"{subset}.json").write_text(json.dumps({
            "subset": subset, "record_count": len(records),
            "record_ids": [r.record_id for r in records],
            "series_files": sorted({r.series_file for r in records}),
        }, indent=2), encoding="utf-8")

    files = sorted(p.name for p in (Path(a.train_data) / "series").glob("series_*.json"))
    random.Random(a.seed).shuffle(files)
    cut = int(0.95 * len(files))
    split = {"seed": a.seed, "train_ratio": 0.95,
             "train_series": files[:cut], "val_series": files[cut:]}
    (out / "phase2_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps({"paper140": len(load_axis_records(a.test_data, "paper140")),
                      "full": len(load_axis_records(a.test_data, "full")),
                      "train_series": cut, "val_series": len(files)-cut}))


if __name__ == "__main__":
    main()
