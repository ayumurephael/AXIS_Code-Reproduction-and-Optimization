from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


EXPECTED_MODEL = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "text_config.hidden_size": 5120,
    "text_config.intermediate_size": 25600,
    "text_config.num_hidden_layers": 64,
    "text_config.num_attention_heads": 64,
    "text_config.num_key_value_heads": 8,
    "text_config.head_dim": 128,
    "text_config.vocab_size": 151936,
    "vision_config.hidden_size": 1152,
    "vision_config.depth": 27,
    "vision_config.num_heads": 16,
    "vision_config.out_hidden_size": 5120,
    "vision_config.patch_size": 16,
    "vision_config.spatial_merge_size": 2,
}


def nested(payload, dotted):
    value = payload
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only hardware/runtime gate for Multi-AXIS-32B."
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    args = parser.parse_args()

    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    versions = {}
    for name in ("torch", "transformers", "flash_attn", "accelerate"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
            check(f"import_{name}", True, versions[name])
        except Exception as exc:  # surfaced in the JSON audit
            check(f"import_{name}", False, f"{type(exc).__name__}: {exc}")

    import torch

    check("cuda_available", torch.cuda.is_available(), torch.version.cuda)
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_gib": total_bytes / 2**30,
                    "free_gib": free_bytes / 2**30,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
        check("all_gpus_bf16", all(item["bf16_supported"] for item in gpus), gpus)
        # Loading a replicated 32B BF16 backbone on an 80 GB GPU is not declared
        # safe here: only a real optimizer-step benchmark can establish that.
        check("gpu_inventory_nonempty", bool(gpus), gpus)

    work_dir = args.work_dir.resolve()
    usage = shutil.disk_usage(work_dir)
    free_gib = usage.free / 2**30
    check(
        "work_disk_free",
        free_gib >= args.minimum_free_gib,
        {"path": str(work_dir), "free_gib": free_gib, "required_gib": args.minimum_free_gib},
    )

    if args.model_dir:
        model_dir = args.model_dir.resolve()
        config_path = model_dir / "config.json"
        index_path = model_dir / "model.safetensors.index.json"
        check("model_config_exists", config_path.is_file(), str(config_path))
        check("model_index_exists", index_path.is_file(), str(index_path))
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mismatches = {
                key: {"expected": expected, "observed": nested(config, key)}
                for key, expected in EXPECTED_MODEL.items()
                if nested(config, key) != expected
            }
            check("model_identity", not mismatches, mismatches)
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index.get("weight_map", {}).values()))
            missing = [name for name in shard_names if not (model_dir / name).is_file()]
            shard_bytes = sum(
                (model_dir / name).stat().st_size for name in shard_names if (model_dir / name).is_file()
            )
            check(
                "model_shards_complete",
                len(shard_names) == 14 and not missing,
                {"shards": len(shard_names), "missing": missing, "bytes": shard_bytes},
            )

    payload = {
        "passed": all(item["passed"] for item in checks),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "versions": versions,
        "gpus": gpus,
        "checks": checks,
        "next_gate": "Run a distinct 2-5 optimizer-step benchmark before formal training.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
