import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


DIR_RE = re.compile(r"^teacher_ts_varfeat20k_(part\d\d)_(regular|hard)_1000_(\d{4}_\d{4})_0621ta$")


def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def copy_file(src: Path, dst: Path, retries: int = 3, sleep_s: float = 1.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(retries):
        try:
            shutil.copy2(src, dst)
            return
        except OSError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(sleep_s)
    raise last_exc


def sync_dir(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.rglob("*"):
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=r"E:\multiaxis\teacheranswer",
        help="Root containing teacher_ts_varfeat20k_*_0621ta directories.",
    )
    parser.add_argument(
        "--target-root",
        default=r"E:\multiaxis\teacheranswer\teacheranswer_train",
        help="Root to store synced training shard directories.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    rows = []
    total_lines = 0
    copied_dirs = 0
    skipped_dirs = 0

    for src_dir in sorted(source_root.iterdir()):
        if not src_dir.is_dir():
            continue
        if src_dir.resolve() == target_root.resolve():
            continue
        match = DIR_RE.match(src_dir.name)
        if not match:
            continue

        answers_path = src_dir / "teacher_gpt55.answers.jsonl"
        if not answers_path.exists():
            skipped_dirs += 1
            continue

        try:
            line_count = count_lines(answers_path)
        except OSError:
            skipped_dirs += 1
            continue

        if line_count <= 0:
            skipped_dirs += 1
            continue

        dst_dir = target_root / src_dir.name
        sync_dir(src_dir, dst_dir)
        copied_dirs += 1
        total_lines += line_count

        rows.append(
            {
                "name": src_dir.name,
                "part": match.group(1),
                "bank": match.group(2),
                "segment": match.group(3),
                "line_count": line_count,
                "complete": line_count == 1000,
                "source": str(src_dir),
                "target": str(dst_dir),
            }
        )

    manifest_tsv = target_root / "manifest.tsv"
    with manifest_tsv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("name\tpart\tbank\tsegment\tline_count\tcomplete\tsource\ttarget\n")
        for row in rows:
            handle.write(
                "{name}\t{part}\t{bank}\t{segment}\t{line_count}\t{complete}\t{source}\t{target}\n".format(
                    **row
                )
            )

    summary = {
        "synced_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "copied_dirs": copied_dirs,
        "skipped_dirs": skipped_dirs,
        "total_lines": total_lines,
        "complete_dirs": sum(1 for row in rows if row["complete"]),
        "partial_dirs": sum(1 for row in rows if not row["complete"]),
    }
    (target_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
