"""Create a publishable AXIS result tree without remote host paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REMOTE_HOME = "/home2/xuhaijie"
REMOTE_REPLACEMENT = "<remote-root>"
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:GEMINI|DEEPSEEK)_API_KEY\s*="),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--raw", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    raw = args.raw.resolve()
    output = args.output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("publishable output must not be inside the raw tree")
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads(
        (raw / "download_manifest.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True)
    rows = []
    total_replacements = 0
    for source_row in source_manifest["files"]:
        relative = Path(source_row["path"])
        source = raw / relative
        data = source.read_bytes()
        assert len(data) == source_row["bytes"]
        assert sha256_bytes(data) == source_row["sha256"]
        text = data.decode("utf-8")
        replacements = text.count(REMOTE_HOME)
        sanitized = text.replace(REMOTE_HOME, REMOTE_REPLACEMENT)
        if any(pattern.search(sanitized) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"secret-like content in {relative.as_posix()}")
        target_data = sanitized.encode("utf-8")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(target_data)
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": len(target_data),
                "sha256": sha256_bytes(target_data),
                "source_bytes": len(data),
                "source_sha256": source_row["sha256"],
                "remote_path_replacements": replacements,
            }
        )
        total_replacements += replacements

    artifact_manifest = {
        "schema_version": 1,
        "experiment": "loss_e2e_0723",
        "purpose": "publishable_result_evidence",
        "sanitization": {
            "remote_home_paths_replaced": total_replacements,
            "replacement_token": REMOTE_REPLACEMENT,
            "api_keys_included": False,
            "model_checkpoints_included": False,
            "pid_files_included": False,
        },
        "source_download_manifest_sha256": sha256_bytes(
            (raw / "download_manifest.json").read_bytes()
        ),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "file_count": artifact_manifest["file_count"],
        "total_bytes": artifact_manifest["total_bytes"],
        "remote_path_replacements": total_replacements,
        "artifact_manifest_sha256": sha256_bytes(
            (output / "artifact_manifest.json").read_bytes()
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
