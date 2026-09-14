from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.utils import save_json, write_jsonl


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _with_order_metadata(
    row: Dict[str, Any],
    *,
    global_index: int,
    shard_index: int,
    index_in_shard: int,
) -> Dict[str, Any]:
    out = dict(row)
    out["global_sample_index"] = int(global_index)
    out["shard_index"] = int(shard_index)
    out["index_in_shard"] = int(index_in_shard)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack one large ts-only JSONL into fixed-size shards.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--prefix", default="raw_series")
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    args = parser.parse_args()

    source_path = _resolve(args.source)
    output_dir = _resolve(args.output_dir)
    shards_dir = output_dir / "shards"
    manifest_path = output_dir / "shards_manifest.json"
    summary_path = output_dir / "shards_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_path)
    shard_size = max(1, int(args.shard_size))
    progress = ProgressPrinter(
        len(rows),
        label="ts-shard",
        step_percent=float(args.progress_step_percent),
    )
    progress.start(extra="written=0")

    manifest: List[Dict[str, Any]] = []
    written = 0
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        end = min(start + shard_size, len(rows))
        shard_rows = [
            _with_order_metadata(
                rows[global_index],
                global_index=global_index,
                shard_index=shard_index,
                index_in_shard=global_index - start,
            )
            for global_index in range(start, end)
        ]
        shard_path = shards_dir / f"{args.prefix}_{shard_index:04d}.jsonl"
        write_jsonl(shard_rows, shard_path)
        manifest.append(
            {
                "shard_index": int(shard_index),
                "path": str(shard_path),
                "start_global_index": int(start),
                "end_global_index_exclusive": int(end),
                "num_rows": int(end - start),
            }
        )
        written = end
        progress.update(written, extra=f"written={written}")

    save_json({"shards": manifest}, manifest_path)
    summary = {
        "source": str(source_path),
        "output_dir": str(output_dir),
        "shards_dir": str(shards_dir),
        "manifest_path": str(manifest_path),
        "num_rows": len(rows),
        "shard_size": shard_size,
        "num_shards": len(manifest),
        "prefix": str(args.prefix),
    }
    save_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
