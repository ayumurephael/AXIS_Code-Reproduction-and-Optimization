"""Fail-closed, resumable post-training pipeline for the AXIS full variant.

The runner waits for an already-running Phase-II job, audits and strips every
epoch checkpoint, validates all epochs, selects the strict minimum validation
NLL, performs the fixed test protocol, and builds a non-secret artifact bundle.
It is intentionally single-host and pins every CUDA subprocess to GPUs 0,1,2.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_EPOCH_STEPS = {1: 9500, 2: 19000, 3: 28500}
EXPECTED_VALIDATION_ROWS = 3000
EXPECTED_FULL_ROWS = 284
EXPECTED_PAPER_ROWS = 140
EXPECTED_GEVAL_ROWS = 335
FATAL_TRAINING_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|"
    r"ChildFailedError|ProcessExitedException|NCCL.{0,80}(?:error|fail)|"
    r"(?:^|[\s:,])nan(?:[\s,}]|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)


class PipelineError(RuntimeError):
    """A fail-closed pipeline invariant was violated."""


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def strict_best_epoch(selection: dict) -> int:
    candidates = list(selection.get("candidates", []))
    if len(candidates) != 3:
        raise PipelineError("checkpoint selection must contain exactly three epochs")
    losses = [float(row["mean_loss"]) for row in candidates]
    best = min(losses)
    if sum(value == best for value in losses) != 1:
        raise PipelineError("validation NLL minimum is not strict")
    selected = candidates[losses.index(best)]
    match = re.search(r"epoch_(\d+)", str(selected["predictions"]))
    if not match:
        raise PipelineError("cannot recover selected epoch from validation path")
    epoch = int(match.group(1))
    if epoch not in EXPECTED_EPOCH_STEPS:
        raise PipelineError(f"invalid selected epoch: {epoch}")
    return epoch


def training_log_failure(text: str) -> str | None:
    match = FATAL_TRAINING_PATTERN.search(text)
    return match.group(0) if match else None


def process_matches(pid: int, marker: str) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    command_path = Path(f"/proc/{pid}/cmdline")
    if command_path.exists():
        command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return marker in command
    return True


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.root = Path(args.root).resolve()
        self.run_dir = (self.root / args.run_dir).resolve()
        self.checkpoint_dir = self.run_dir / "formal" / "checkpoints"
        self.training_log = self.run_dir / "formal" / "phase2.log"
        self.training_pid_file = self.run_dir / "formal" / "phase2.pid"
        self.pipeline_dir = self.run_dir / "posttrain_pipeline"
        self.events = self.pipeline_dir / "events.jsonl"
        self.status_path = self.pipeline_dir / "status.json"
        self.model_name = str(Path(args.model_name).resolve())
        self.credential_file = Path(args.credential_file).resolve()
        self.poll_seconds = args.poll_seconds
        # Preserve the virtual-environment entry point. Resolving this symlink
        # selects the base interpreter and silently drops the venv site-packages.
        self.python = str(self.root / ".venv" / "bin" / "python")
        self.gpus = args.gpus
        self.expected_donor_top_m = args.expected_donor_top_m
        self.stop_before_geval = bool(args.stop_before_geval)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

    def event(self, stage: str, state: str, **details) -> None:
        payload = {"time": timestamp(), "stage": stage, "state": state, **details}
        append_jsonl(self.events, payload)
        atomic_json(self.status_path, payload)

    def marker(self, stage: str) -> Path:
        return self.pipeline_dir / f"{stage}.done.json"

    def complete(self, stage: str, **details) -> None:
        payload = {"time": timestamp(), "stage": stage, "ok": True, **details}
        atomic_json(self.marker(stage), payload)
        self.event(stage, "completed", **details)

    def base_env(self, cuda_visible_devices: str) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "AXIS_MODEL_NAME": self.model_name,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        })
        return environment

    def run_command(
        self,
        stage: str,
        command: Sequence[str],
        log_path: Path,
        *,
        cuda_visible_devices: str,
        append: bool = True,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.event(stage, "running", command=list(command), log=str(log_path))
        with log_path.open("a" if append else "w", encoding="utf-8") as handle:
            if append:
                handle.write(json.dumps({"time": timestamp(), "command": list(command)}) + "\n")
                handle.flush()
            result = subprocess.run(
                list(command),
                cwd=self.root,
                env=self.base_env(cuda_visible_devices),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            self.event(stage, "failed", returncode=result.returncode, log=str(log_path))
            raise PipelineError(f"{stage} failed with exit code {result.returncode}")

    def wait_for_training(self) -> None:
        stage = "wait_training"
        if self.marker(stage).exists():
            return
        if not self.training_pid_file.exists():
            raise PipelineError(f"missing training PID file: {self.training_pid_file}")
        pid = int(self.training_pid_file.read_text(encoding="utf-8").strip())
        self.event(stage, "waiting", pid=pid)
        last_notice = 0.0
        while process_matches(pid, "train_phase2_architecture_redesign"):
            now = time.monotonic()
            if now - last_notice >= 300:
                self.event(stage, "waiting", pid=pid)
                last_notice = now
            time.sleep(self.poll_seconds)
        time.sleep(5)
        if not self.training_log.exists():
            raise PipelineError("training process exited without a log")
        failure = training_log_failure(self.training_log.read_text(encoding="utf-8", errors="replace"))
        if failure:
            raise PipelineError(f"fatal signature in training log: {failure}")
        runtime_path = self.checkpoint_dir / "runtime.json"
        if not runtime_path.exists():
            raise PipelineError("training process exited without runtime.json")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if int(runtime.get("steps", -1)) != 28500 or int(runtime.get("world_size", -1)) != 3:
            raise PipelineError(f"training runtime mismatch: {runtime}")
        missing = [
            str(self.checkpoint_dir / f"epoch_{epoch}.pth")
            for epoch in EXPECTED_EPOCH_STEPS
            if not (self.checkpoint_dir / f"epoch_{epoch}.pth").is_file()
        ]
        if missing:
            raise PipelineError(f"training completed without epoch checkpoints: {missing}")
        self.complete(stage, pid=pid, runtime=runtime)

    def audit_and_strip_checkpoints(self) -> None:
        stage = "checkpoint_audit_and_strip"
        if self.marker(stage).exists():
            return
        self.event(stage, "running")
        import torch

        from .architecture_redesign import audit_architecture_checkpoint
        from .audit_loss_objective_checkpoint import audit_loss_objective_checkpoint
        from .strip_checkpoints import strip_checkpoint_payload

        reports = []
        for epoch, expected_step in EXPECTED_EPOCH_STEPS.items():
            source = self.checkpoint_dir / f"epoch_{epoch}.pth"
            target = self.checkpoint_dir / f"epoch_{epoch}_inference.pth"
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", -1)) != epoch:
                raise PipelineError(f"epoch metadata mismatch in {source}")
            if int(payload.get("global_step", -1)) != expected_step:
                raise PipelineError(f"global step mismatch in {source}")
            if "optimizer_state_dict" not in payload:
                raise PipelineError(f"formal checkpoint is missing optimizer state: {source}")
            loss_report = audit_loss_objective_checkpoint(
                payload, expected_donor_top_m=self.expected_donor_top_m
            )
            architecture_report = audit_architecture_checkpoint(payload, expected_variant="full")
            if int(architecture_report.get("qk_norm_seq_len", -1)) != 40:
                raise PipelineError("QK-Norm sequence length is not 40")
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save(strip_checkpoint_payload(payload), temporary)
            os.replace(temporary, target)
            del payload
            gc.collect()

            stripped = torch.load(target, map_location="cpu", weights_only=False)
            if "optimizer_state_dict" in stripped:
                raise PipelineError(f"optimizer state survived stripping: {target}")
            stripped_loss = audit_loss_objective_checkpoint(
                stripped, expected_donor_top_m=self.expected_donor_top_m
            )
            stripped_architecture = audit_architecture_checkpoint(
                stripped, expected_variant="full"
            )
            report = {
                "epoch": epoch,
                "global_step": expected_step,
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "inference_checkpoint": str(target),
                "inference_bytes": target.stat().st_size,
                "loss_audit": loss_report,
                "architecture_audit": architecture_report,
                "stripped_loss_audit": stripped_loss,
                "stripped_architecture_audit": stripped_architecture,
            }
            atomic_json(self.pipeline_dir / f"epoch_{epoch}_checkpoint_audit.json", report)
            reports.append(report)
            del stripped
            gc.collect()
        self.complete(stage, epochs=len(reports))

    def torchrun_command(self, checkpoint: Path, data: Path, output: Path) -> list[str]:
        return [
            self.python, "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=3", "-m", "tools.axis_repro.run_inference_cli",
            "--checkpoint", str(checkpoint), "--data", str(data),
            "--subset", "full", "--modes", "base", "--output", str(output),
            "--architecture-variant", "full", "--qk-norm-seq-len", "40",
            "--gate-bias", "-2",
        ]

    def run_validation(self, epoch: int) -> None:
        stage = f"validation_epoch_{epoch}"
        output = self.run_dir / "validation" / f"epoch_{epoch}"
        predictions = output / "predictions.jsonl"
        audit_path = output / "audit_validation.json"
        if self.marker(stage).exists() and predictions.exists() and audit_path.exists():
            return
        checkpoint = self.checkpoint_dir / f"epoch_{epoch}_inference.pth"
        command = self.torchrun_command(
            checkpoint, self.root / "data" / "anomaly_llava_training_dataset", output
        )
        command.extend([
            "--series-split-manifest",
            str(self.root / "experiments" / "reproduction" / "manifests" / "phase2_split.json"),
            "--series-split-key", "val_series",
        ])
        self.run_command(
            stage, command, output / "inference.log", cuda_visible_devices=self.gpus
        )
        audit_command = [
            self.python, "-m", "tools.axis_repro.audit_validation",
            "--predictions", str(predictions),
            "--series-manifest",
            str(self.root / "experiments" / "reproduction" / "manifests" / "phase2_split.json"),
            "--series-keys", "val_series",
        ]
        self.run_command(
            stage, audit_command, audit_path, cuda_visible_devices="", append=False
        )
        rows = read_jsonl(predictions)
        if len(rows) != EXPECTED_VALIDATION_ROWS or len({x["series_file"] for x in rows}) != 1500:
            raise PipelineError(f"validation epoch {epoch} coverage mismatch")
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("architecture_variant") != "full" or int(manifest.get("qk_norm_seq_len", -1)) != 40:
            raise PipelineError(f"validation epoch {epoch} architecture mismatch")
        self.complete(stage, rows=len(rows), series=1500)

    def select_checkpoint(self) -> int:
        stage = "select_best_validation_nll"
        selection_path = self.run_dir / "best_validation_loss.json"
        selected_path = self.run_dir / "selected_checkpoint.json"
        if self.marker(stage).exists() and selected_path.exists():
            return int(json.loads(selected_path.read_text(encoding="utf-8"))["epoch"])
        predictions = [
            self.run_dir / "validation" / f"epoch_{epoch}" / "predictions.jsonl"
            for epoch in EXPECTED_EPOCH_STEPS
        ]
        command = [
            self.python, "-m", "tools.axis_repro.select_best_loss",
            *(str(path) for path in predictions), "--output", str(selection_path),
        ]
        self.run_command(
            stage, command, self.pipeline_dir / "select_best.log", cuda_visible_devices=""
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        epoch = strict_best_epoch(selection)
        checkpoint = self.checkpoint_dir / f"epoch_{epoch}_inference.pth"
        atomic_json(selected_path, {
            "selection_metric": selection["selection_metric"],
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "mean_validation_nll": float(selection["best"]["mean_loss"]),
        })
        self.complete(stage, epoch=epoch, mean_validation_nll=selection["best"]["mean_loss"])
        return epoch

    def run_full284(self, epoch: int) -> Path:
        stage = "full284_inference"
        output = self.run_dir / f"test_epoch_{epoch}_full284"
        predictions = output / "predictions.jsonl"
        if not self.marker(stage).exists():
            command = self.torchrun_command(
                self.checkpoint_dir / f"epoch_{epoch}_inference.pth",
                self.root / "data" / "AXIS_qa_test",
                output,
            )
            self.run_command(
                stage, command, output / "inference.log", cuda_visible_devices=self.gpus
            )
            audit_command = [
                self.python, "-m", "tools.axis_repro.audit_results",
                "--predictions", str(predictions), "--manifest",
                str(self.root / "experiments" / "reproduction" / "manifests" / "full.json"),
                "--modes", "base",
            ]
            self.run_command(
                stage, audit_command, output / "audit_results.json",
                cuda_visible_devices="", append=False,
            )
            rows = read_jsonl(predictions)
            if len(rows) != EXPECTED_FULL_ROWS:
                raise PipelineError(f"full284 row mismatch: {len(rows)}")
            self.complete(stage, rows=len(rows), epoch=epoch)
        return predictions

    def filter_paper140(self, epoch: int, full_predictions: Path) -> Path:
        stage = "paper140_filter"
        output = self.run_dir / f"test_epoch_{epoch}_paper140"
        predictions = output / "predictions.jsonl"
        if not self.marker(stage).exists():
            command = [
                self.python, "-m", "tools.axis_repro.filter_subset",
                "--predictions", str(full_predictions), "--manifest",
                str(self.root / "experiments" / "reproduction" / "manifests" / "paper140.json"),
                "--output", str(predictions), "--modes", "base",
            ]
            self.run_command(
                stage, command, output / "filter.log", cuda_visible_devices=""
            )
            audit_command = [
                self.python, "-m", "tools.axis_repro.audit_results",
                "--predictions", str(predictions), "--manifest",
                str(self.root / "experiments" / "reproduction" / "manifests" / "paper140.json"),
                "--modes", "base",
            ]
            self.run_command(
                stage, audit_command, output / "audit_predictions.json",
                cuda_visible_devices="", append=False,
            )
            rows = read_jsonl(predictions)
            if len(rows) != EXPECTED_PAPER_ROWS:
                raise PipelineError(f"paper140 row mismatch: {len(rows)}")
            self.complete(stage, rows=len(rows), epoch=epoch)
        return predictions

    def system_ca(self) -> Path:
        candidates = [
            Path("/etc/pki/tls/certs/ca-bundle.crt"),
            Path("/etc/ssl/certs/ca-certificates.crt"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise PipelineError("system CA bundle was not found")

    def run_geval(self, epoch: int, predictions: Path) -> Path:
        stage = "deepseek_v4_pro_geval"
        output_dir = predictions.parent
        scores = output_dir / "geval.jsonl"
        if self.marker(stage).exists() and scores.exists():
            return scores
        if not self.credential_file.is_file():
            raise PipelineError("protected DeepSeek credential file is missing")
        if stat.S_IMODE(self.credential_file.stat().st_mode) != 0o600:
            raise PipelineError("DeepSeek credential file mode must be 600")
        ca_bundle = self.system_ca()
        environment = self.base_env("0")
        environment["SSL_CERT_FILE"] = str(ca_bundle)
        environment["REQUESTS_CA_BUNDLE"] = str(ca_bundle)
        command = [
            self.python, "-m", "tools.axis_repro.run_with_secret",
            "--credential-file", str(self.credential_file), "--",
            self.python, "-m", "tools.axis_repro.geval_resilient",
            "--predictions", str(predictions), "--output", str(scores),
            "--model", "deepseek-v4-pro", "--fallback-samples", "20",
            "--max-tokens", "4096", "--primary-workers", "8",
            "--fallback-workers", "1",
        ]
        log_path = output_dir / "geval.log"
        self.event(stage, "running", log=str(log_path), gpu_server=True)
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                command, cwd=self.root, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            self.event(stage, "failed", returncode=result.returncode)
            raise PipelineError(f"G-Eval failed with exit code {result.returncode}")
        audit_command = [
            self.python, "-m", "tools.axis_repro.audit_results",
            "--predictions", str(predictions), "--scores", str(scores),
            "--manifest",
            str(self.root / "experiments" / "reproduction" / "manifests" / "paper140.json"),
            "--modes", "base",
        ]
        self.run_command(
            stage, audit_command, output_dir / "audit_geval.json",
            cuda_visible_devices="", append=False,
        )
        score_rows = read_jsonl(scores)
        if len(score_rows) != EXPECTED_GEVAL_ROWS:
            raise PipelineError(f"G-Eval row mismatch: {len(score_rows)}")
        self.complete(stage, score_rows=len(score_rows), epoch=epoch)
        return scores

    def aggregate_and_bundle(self, epoch: int, scores: Path) -> None:
        stage = "aggregate_and_bundle"
        if self.marker(stage).exists():
            return
        output_dir = scores.parent
        table_prefix = output_dir / "table1"
        command = [
            self.python, "-m", "tools.axis_repro.build_tables",
            "--scores", str(scores), "--output-prefix", str(table_prefix),
        ]
        self.run_command(
            stage, command, output_dir / "aggregate.log", cuda_visible_devices=""
        )
        table = json.loads(table_prefix.with_suffix(".json").read_text(encoding="utf-8"))
        metrics = table.get("models", {}).get("base", {})
        expected_metrics = {
            "multiple_choice/final", "multiple_choice/correctness",
            "multiple_choice/reasoning_quality", "open_ended/final",
            "open_ended/accuracy", "open_ended/completeness",
            "open_ended/relevance", "true_false/final",
            "true_false/correctness", "true_false/justification_quality",
        }
        if not expected_metrics.issubset(metrics):
            raise PipelineError("Table 1 is missing one or more of the ten required metrics")

        artifact_paths: list[Path] = [
            self.run_dir / "index" / "counterfactual_index_audit.json",
            self.checkpoint_dir / "runtime.json",
            self.training_log,
            self.run_dir / "best_validation_loss.json",
            self.run_dir / "selected_checkpoint.json",
            scores,
            output_dir / "audit_geval.json",
            table_prefix.with_suffix(".json"),
            table_prefix.with_suffix(".md"),
        ]
        for epoch_index in EXPECTED_EPOCH_STEPS:
            artifact_paths.extend([
                self.pipeline_dir / f"epoch_{epoch_index}_checkpoint_audit.json",
                self.run_dir / "validation" / f"epoch_{epoch_index}" / "predictions.jsonl",
                self.run_dir / "validation" / f"epoch_{epoch_index}" / "run_manifest.json",
                self.run_dir / "validation" / f"epoch_{epoch_index}" / "audit_validation.json",
            ])
        full_dir = self.run_dir / f"test_epoch_{epoch}_full284"
        paper_dir = self.run_dir / f"test_epoch_{epoch}_paper140"
        artifact_paths.extend([
            full_dir / "predictions.jsonl", full_dir / "run_manifest.json",
            full_dir / "audit_results.json", paper_dir / "predictions.jsonl",
            paper_dir / "audit_predictions.json",
        ])
        missing = [str(path) for path in artifact_paths if not path.is_file()]
        if missing:
            raise PipelineError(f"artifact bundle inputs are missing: {missing}")
        manifest = {
            "created_at": timestamp(),
            "selected_epoch": epoch,
            "metrics": {key: metrics[key] for key in sorted(expected_metrics)},
            "files": {
                str(path.relative_to(self.root)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            },
        }
        manifest_path = self.run_dir / "artifacts_manifest.json"
        atomic_json(manifest_path, manifest)
        bundle_path = self.run_dir / "architecture_argmin_v3_nonsecret_artifacts.tar.gz"
        temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
        with tarfile.open(temporary, "w:gz") as archive:
            for path in [*artifact_paths, manifest_path]:
                archive.add(path, arcname=str(path.relative_to(self.root)), recursive=False)
        os.replace(temporary, bundle_path)
        self.complete(
            stage, selected_epoch=epoch, bundle=str(bundle_path),
            bundle_bytes=bundle_path.stat().st_size,
            bundle_sha256=sha256_file(bundle_path), metrics=manifest["metrics"],
        )

    def run(self) -> None:
        try:
            self.wait_for_training()
            self.audit_and_strip_checkpoints()
            for epoch in EXPECTED_EPOCH_STEPS:
                self.run_validation(epoch)
            selected_epoch = self.select_checkpoint()
            full_predictions = self.run_full284(selected_epoch)
            paper_predictions = self.filter_paper140(selected_epoch, full_predictions)
            if self.stop_before_geval:
                self.event(
                    "pipeline",
                    "awaiting_geval_approval",
                    selected_epoch=selected_epoch,
                    paper140_predictions=str(paper_predictions),
                )
                return
            scores = self.run_geval(selected_epoch, paper_predictions)
            self.aggregate_and_bundle(selected_epoch, scores)
            self.event("pipeline", "completed", selected_epoch=selected_epoch)
        except Exception as exc:
            self.event("pipeline", "failed", error_type=type(exc).__name__, error=str(exc))
            raise


def acquire_lock(path: Path):
    """Hold an exclusive Linux advisory lock for the lifetime of the pipeline."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PipelineError("another post-training pipeline is already active") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--run-dir", default="experiments/reproduction/architecture_argmin_v3"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--credential-file", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--expected-donor-top-m", type=int, default=1)
    parser.add_argument(
        "--stop-before-geval", action="store_true",
        help="Complete all local/GPU inference stages, then stop before external API transfer.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gpus != "0,1,2":
        raise PipelineError("formal pipeline must use exactly GPUs 0,1,2")
    pipeline = Pipeline(args)
    lock = acquire_lock(pipeline.pipeline_dir / "pipeline.lock")
    try:
        pipeline.run()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
