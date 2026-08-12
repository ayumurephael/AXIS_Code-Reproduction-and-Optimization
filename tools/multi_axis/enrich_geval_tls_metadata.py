from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich(
    input_path: Path,
    ca_bundle: Path,
    output_path: Path,
    audit_path: Path,
) -> Dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ so raw scores are preserved")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("Enriched output or audit path already exists")
    if not ca_bundle.is_file():
        raise FileNotFoundError(ca_bundle)
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("G-Eval input is empty")
    ca_sha256 = sha256_file(ca_bundle)
    enriched = []
    for index, row in enumerate(rows):
        if row.get("provider_profile") != "gpt-5.4":
            raise ValueError(f"Row {index} is not a gpt-5.4 result")
        if len(str(row.get("endpoint_host_sha256") or "")) != 64:
            raise ValueError(f"Row {index} lacks endpoint provenance")
        existing = row.get("tls_ca_bundle_sha256")
        if existing not in (None, ""):
            raise ValueError(f"Row {index} already has TLS provenance")
        item = dict(row)
        item["tls_ca_bundle_sha256"] = ca_sha256
        enriched.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, enriched)
    for raw, item in zip(rows, read_jsonl(output_path)):
        recovered = dict(item)
        assert recovered.pop("tls_ca_bundle_sha256") == ca_sha256
        if recovered != raw:
            raise AssertionError("Metadata enrichment changed a score record")
    audit = {
        "protocol": "gpt54-tls-provenance-enrichment-v1",
        "rows": len(rows),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "ca_bundle_sha256": ca_sha256,
        "only_added_field": "tls_ca_bundle_sha256",
        "scores_changed": False,
        "passed": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add previously omitted GPT-5.4 TLS provenance without changing scores."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = enrich(args.input, args.ca_bundle, args.output, args.audit_output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
