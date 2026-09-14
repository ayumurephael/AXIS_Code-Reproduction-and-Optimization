from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _summary_matches(summary_path: Path, *, num_samples: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("num_samples") or -1) == int(num_samples)


def _run(command: list[str], *, pythonpath_entry: Path | None = None) -> None:
    print(json.dumps({"status": "running", "command": command}, ensure_ascii=False), flush=True)
    env = os.environ.copy()
    if pythonpath_entry is not None:
        existing = env.get("PYTHONPATH", "")
        entries = [str(pythonpath_entry)]
        if existing:
            entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(entries)
    subprocess.run(command, check=True, env=env)


def _resolve_script(script_dir: Path, name: str) -> Path:
    candidate = script_dir / name
    if candidate.exists():
        return candidate
    bootstrap_dir = Path(__import__("_bootstrap").__file__).resolve().parent
    fallback = bootstrap_dir / name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Unable to locate required helper script: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate variable-channel tsdata and highlighted windows in one output directory."
    )
    parser.add_argument("--legacy-src", default=None, help="Optional legacy TSAD_dataset_gen-axis directory.")
    parser.add_argument("--output-dir", required=True, help="Target folder that will contain raw_series, shards, images, and windows jsonl.")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--seq-len-min", type=int, default=128)
    parser.add_argument("--seq-len-max", type=int, default=512)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-features-min", type=int, default=5)
    parser.add_argument("--num-features-max", type=int, default=50)
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--samples-per-series", type=int, default=2)
    parser.add_argument("--min-window-size", type=int, default=15)
    parser.add_argument("--max-window-size", type=int, default=60)
    parser.add_argument("--window-anomaly-ratio", type=float, default=0.5)
    parser.add_argument("--sample-id-prefix", default=None)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--progress-step-percent", type=float, default=1.0)
    parser.add_argument(
        "--force-regenerate-raw",
        action="store_true",
        help="Regenerate raw_series even if dataset_summary.json already matches the requested sample count.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    bootstrap_dir = Path(__import__("_bootstrap").__file__).resolve().parent
    ts_script = _resolve_script(script_dir, "mvaxis_generate_ts_only.py")
    window_script = _resolve_script(script_dir, "mvaxis_sample_windows_to_images.py")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summary_path = output_dir / "dataset_summary.json"
    should_generate_raw = bool(args.force_regenerate_raw) or not _summary_matches(
        raw_summary_path,
        num_samples=int(args.num_samples),
    )

    if should_generate_raw:
        ts_command = [
            sys.executable,
            str(ts_script),
            "--output-dir",
            str(output_dir),
            "--num-samples",
            str(int(args.num_samples)),
            "--seq-len",
            str(int(args.seq_len)),
            "--seq-len-min",
            str(int(args.seq_len_min)),
            "--seq-len-max",
            str(int(args.seq_len_max)),
            "--num-features",
            str(int(args.num_features)),
            "--num-features-min",
            str(int(args.num_features_min)),
            "--num-features-max",
            str(int(args.num_features_max)),
            "--anomaly-ratio",
            str(float(args.anomaly_ratio)),
            "--seed",
            str(int(args.seed)),
            "--shard-size",
            str(int(args.shard_size)),
            "--progress-step-percent",
            str(float(args.progress_step_percent)),
        ]
        if args.sample_id_prefix:
            ts_command.extend(["--sample-id-prefix", str(args.sample_id_prefix)])
        if args.legacy_src:
            ts_command.extend(["--legacy-src", str(args.legacy_src)])
        _run(ts_command, pythonpath_entry=bootstrap_dir)
    else:
        print(
            json.dumps(
                {
                    "status": "skip_raw_generation",
                    "dataset_summary": str(raw_summary_path),
                    "reason": "existing dataset_summary.json already matches num_samples",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    window_command = [
        sys.executable,
        str(window_script),
        "--source",
        str(output_dir / "raw_series.jsonl"),
        "--output-dir",
        str(output_dir),
        "--samples-per-series",
        str(int(args.samples_per_series)),
        "--min-window-size",
        str(int(args.min_window_size)),
        "--max-window-size",
        str(int(args.max_window_size)),
        "--anomaly-ratio",
        str(float(args.window_anomaly_ratio)),
        "--seed",
        str(int(args.seed)),
        "--progress-step-percent",
        str(float(args.progress_step_percent)),
    ]
    _run(window_command, pythonpath_entry=bootstrap_dir)


if __name__ == "__main__":
    main()
