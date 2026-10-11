from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.models.MultiAXIS.config import QWEN3_VL_32B_REVISION, QWEN3_VL_MODEL_ID


EXPECTED_BYTES = 66_726_519_322
EXPECTED_WEIGHT_SHARDS = 14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and audit the immutable Multi-AXIS-32B backbone."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(
        QWEN3_VL_MODEL_ID,
        revision=QWEN3_VL_32B_REVISION,
        files_metadata=True,
    )
    if info.sha != QWEN3_VL_32B_REVISION:
        raise RuntimeError(f"Resolved revision changed unexpectedly: {info.sha}")
    remote_bytes = sum(int(item.size or 0) for item in info.siblings)
    if remote_bytes != EXPECTED_BYTES:
        raise RuntimeError(
            f"Pinned snapshot size {remote_bytes} differs from {EXPECTED_BYTES}"
        )
    snapshot_download(
        repo_id=QWEN3_VL_MODEL_ID,
        revision=QWEN3_VL_32B_REVISION,
        local_dir=target,
        max_workers=args.max_workers,
    )

    files = []
    for path in sorted(item for item in target.rglob("*") if item.is_file()):
        relative = path.relative_to(target).as_posix()
        if relative.startswith(".cache/") or relative in {
            "SNAPSHOT_MANIFEST.json",
            "SNAPSHOT_COMPLETE",
        }:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    shards = [item for item in files if item["path"].endswith(".safetensors")]
    actual_bytes = sum(item["bytes"] for item in files)
    if len(shards) != EXPECTED_WEIGHT_SHARDS:
        raise RuntimeError(f"Expected 14 safetensors shards, found {len(shards)}")
    if actual_bytes != EXPECTED_BYTES:
        raise RuntimeError(
            f"Downloaded snapshot size {actual_bytes} differs from {EXPECTED_BYTES}"
        )
    manifest = {
        "format": "multi-axis-vl-32b-snapshot-v1",
        "repo_id": QWEN3_VL_MODEL_ID,
        "revision": QWEN3_VL_32B_REVISION,
        "file_count": len(files),
        "total_bytes": actual_bytes,
        "weight_shards": len(shards),
        "files": files,
    }
    (target / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "SNAPSHOT_COMPLETE").write_text(
        QWEN3_VL_32B_REVISION + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("repo_id", "revision", "file_count", "total_bytes")}
        )
    )


if __name__ == "__main__":
    main()
