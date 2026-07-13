"""Fail-closed coverage audit for merged Phase-II validation predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--series-manifest", required=True)
    parser.add_argument("--series-keys", nargs="+", required=True)
    parser.add_argument("--records-per-series", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(Path(args.series_manifest).read_text(encoding="utf-8"))
    expected = set()
    for key in args.series_keys:
        expected.update(manifest[key])
    rows = read_jsonl(args.predictions)
    keys = [(x["record_id"], x["mode"]) for x in rows]
    series_counts = Counter(x["series_file"] for x in rows)
    actual = set(series_counts)
    errors = []
    if len(keys) != len(set(keys)):
        errors.append("duplicate (record_id, mode)")
    if set(x["mode"] for x in rows) != {"base"}:
        errors.append("validation contains modes other than base")
    if actual != expected:
        errors.append(
            f"series coverage mismatch: got={len(actual)} expected={len(expected)}"
        )
    bad_counts = {k: v for k, v in series_counts.items() if v != args.records_per_series}
    if bad_counts:
        errors.append(f"series with unexpected record count: {len(bad_counts)}")
    if any(not str(x.get("response", "")).strip() for x in rows):
        errors.append("empty response")
    summary = {
        "rows": len(rows),
        "series": len(actual),
        "expected_series": len(expected),
        "records_per_series": args.records_per_series,
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
