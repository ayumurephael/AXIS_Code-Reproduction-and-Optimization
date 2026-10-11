from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download as hf_snapshot_download

from src.models.MultiAXIS.config import QWEN3_VL_32B_REVISION, QWEN3_VL_MODEL_ID


EXPECTED_BYTES = 66_726_519_322
EXPECTED_WEIGHT_SHARDS = 14
MODELSCOPE_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
MODELSCOPE_REVISION = "master"
MODELSCOPE_AUDITED_COMMIT = "8d6f46b05db6abe0cb3a5ca8c817929cc0fa2d5a"
EXPECTED_SHARDS = {
    "model-00001-of-00014.safetensors": (
        4_932_328_944,
        "6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb",
    ),
    "model-00002-of-00014.safetensors": (
        4_875_990_528,
        "d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa",
    ),
    "model-00003-of-00014.safetensors": (
        4_875_990_552,
        "54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940",
    ),
    "model-00004-of-00014.safetensors": (
        4_875_990_584,
        "ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f",
    ),
    "model-00005-of-00014.safetensors": (
        4_875_990_584,
        "fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2",
    ),
    "model-00006-of-00014.safetensors": (
        4_875_990_584,
        "82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974",
    ),
    "model-00007-of-00014.safetensors": (
        4_875_990_584,
        "fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934",
    ),
    "model-00008-of-00014.safetensors": (
        4_875_990_584,
        "431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f",
    ),
    "model-00009-of-00014.safetensors": (
        4_875_990_584,
        "3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47",
    ),
    "model-00010-of-00014.safetensors": (
        4_875_990_584,
        "aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f",
    ),
    "model-00011-of-00014.safetensors": (
        4_875_990_584,
        "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f",
    ),
    "model-00012-of-00014.safetensors": (
        4_875_990_584,
        "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e",
    ),
    "model-00013-of-00014.safetensors": (
        4_875_990_584,
        "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c",
    ),
    "model-00014-of-00014.safetensors": (
        3_270_697_008,
        "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b",
    ),
}


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
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate an already transferred snapshot without contacting a registry.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Select the immutable HF revision or hash-audited Qwen ModelScope mirror.",
    )
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)

    if args.audit_only:
        source_revision = (
            QWEN3_VL_32B_REVISION
            if args.source == "huggingface"
            else MODELSCOPE_AUDITED_COMMIT
        )
    elif args.source == "huggingface":
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
        hf_snapshot_download(
            repo_id=QWEN3_VL_MODEL_ID,
            revision=QWEN3_VL_32B_REVISION,
            local_dir=target,
            max_workers=args.max_workers,
        )
        source_revision = QWEN3_VL_32B_REVISION
    else:
        try:
            from modelscope.hub.snapshot_download import (
                snapshot_download as ms_snapshot_download,
            )
        except ImportError as error:
            raise RuntimeError(
                "ModelScope fallback requires modelscope==1.34.0; "
                "install the optional requirements file."
            ) from error
        ms_snapshot_download(
            model_id=MODELSCOPE_MODEL_ID,
            revision=MODELSCOPE_REVISION,
            local_dir=str(target),
            max_workers=args.max_workers,
        )
        source_revision = MODELSCOPE_AUDITED_COMMIT

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
    shard_index = {item["path"]: item for item in shards}
    for name, (expected_size, expected_sha256) in EXPECTED_SHARDS.items():
        actual = shard_index.get(name)
        if actual is None:
            raise RuntimeError(f"Required audited shard is missing: {name}")
        if actual["bytes"] != expected_size or actual["sha256"] != expected_sha256:
            raise RuntimeError(
                f"Audited shard differs from the canonical 32B snapshot: {name}"
            )
    if actual_bytes != EXPECTED_BYTES:
        raise RuntimeError(
            f"Downloaded snapshot size {actual_bytes} differs from {EXPECTED_BYTES}"
        )
    manifest = {
        "format": "multi-axis-vl-32b-snapshot-v1",
        "repo_id": QWEN3_VL_MODEL_ID,
        "revision": QWEN3_VL_32B_REVISION,
        "download_source": args.source,
        "source_revision": source_revision,
        "modelscope_audited_commit": MODELSCOPE_AUDITED_COMMIT,
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
            {
                key: manifest[key]
                for key in ("repo_id", "revision", "file_count", "total_bytes")
            }
        )
    )


if __name__ == "__main__":
    main()
