from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


MANIFEST_NAME = "DELIVERY_MANIFEST.json"
SUMS_NAME = "SHA256SUMS.txt"
EXCLUDED = {MANIFEST_NAME, SUMS_NAME}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path):
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED or "/.cache/" in f"/{relative}":
            continue
        yield {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def build(root: Path) -> dict:
    files = list(inventory(root))
    payload = {
        "format": "multi-axis-32b-delivery-v1",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    atomic_text(
        root / MANIFEST_NAME,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_text(
        root / SUMS_NAME,
        "".join(f'{item["sha256"]}  {item["path"]}\n' for item in files),
    )
    return payload


def verify(root: Path) -> dict:
    expected = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    observed = list(inventory(root))
    expected_by_path = {item["path"]: item for item in expected["files"]}
    observed_by_path = {item["path"]: item for item in observed}
    missing = sorted(set(expected_by_path) - set(observed_by_path))
    unexpected = sorted(set(observed_by_path) - set(expected_by_path))
    mismatched = sorted(
        path
        for path in set(expected_by_path) & set(observed_by_path)
        if expected_by_path[path] != observed_by_path[path]
    )
    passed = (
        expected.get("format") == "multi-axis-32b-delivery-v1"
        and not missing
        and not unexpected
        and not mismatched
        and expected.get("file_count") == len(observed)
        and expected.get("total_bytes") == sum(item["bytes"] for item in observed)
    )
    return {
        "passed": passed,
        "expected_files": expected.get("file_count"),
        "observed_files": len(observed),
        "expected_bytes": expected.get("total_bytes"),
        "observed_bytes": sum(item["bytes"] for item in observed),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    payload = build(root) if args.action == "build" else verify(root)
    print(json.dumps(payload if args.action == "verify" else {
        "file_count": payload["file_count"], "total_bytes": payload["total_bytes"]
    }, ensure_ascii=False, indent=2))
    if args.action == "verify" and not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
