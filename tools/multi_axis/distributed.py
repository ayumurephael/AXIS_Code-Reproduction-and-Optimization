from __future__ import annotations

import os
import socket
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import torch
import torch.distributed as dist


def validate_topology_records(
    records: Sequence[Dict[str, Any]],
    *,
    world_size: int,
    expected_nodes: int,
    required_gpu_substring: str | None = None,
) -> Dict[str, Any]:
    """Validate a global rank-to-node/GPU mapping and return an audit summary."""

    if expected_nodes <= 0 or world_size % expected_nodes:
        raise RuntimeError(
            f"world_size={world_size} is not divisible by expected_nodes={expected_nodes}"
        )
    if len(records) != world_size:
        raise RuntimeError(
            f"Topology gathered {len(records)} rank records for world_size={world_size}"
        )
    ordered = sorted(records, key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in ordered]
    if ranks != list(range(world_size)):
        raise RuntimeError(f"Distributed ranks are incomplete or duplicated: {ranks}")

    by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    device_keys = set()
    for item in ordered:
        hostname = str(item["hostname"])
        local_rank = int(item["local_rank"])
        device_key = (hostname, local_rank)
        if device_key in device_keys:
            raise RuntimeError(f"Duplicate distributed device assignment: {device_key}")
        device_keys.add(device_key)
        if required_gpu_substring and required_gpu_substring not in str(item["gpu_name"]):
            raise RuntimeError(
                f"Rank {item['rank']} uses {item['gpu_name']!r}, expected "
                f"a GPU containing {required_gpu_substring!r}"
            )
        by_node[hostname].append(dict(item))

    if len(by_node) != expected_nodes:
        raise RuntimeError(
            f"Expected {expected_nodes} distributed nodes, got {sorted(by_node)}"
        )
    gpus_per_node = world_size // expected_nodes
    nodes = []
    for hostname, node_records in sorted(by_node.items()):
        local_ranks = sorted(int(item["local_rank"]) for item in node_records)
        if local_ranks != list(range(gpus_per_node)):
            raise RuntimeError(
                f"Node {hostname} local ranks are {local_ranks}, expected "
                f"0..{gpus_per_node - 1}"
            )
        visible_counts = {int(item["local_cuda_device_count"]) for item in node_records}
        if visible_counts != {gpus_per_node}:
            raise RuntimeError(
                f"Node {hostname} exposes local CUDA counts {sorted(visible_counts)}, "
                f"expected exactly {gpus_per_node}"
            )
        nodes.append(
            {
                "hostname": hostname,
                "ranks": sorted(int(item["rank"]) for item in node_records),
                "local_ranks": local_ranks,
                "gpu_names": [
                    str(item["gpu_name"])
                    for item in sorted(node_records, key=lambda value: int(value["local_rank"]))
                ],
            }
        )
    return {
        "world_size": world_size,
        "node_count": expected_nodes,
        "gpus_per_node": gpus_per_node,
        "nodes": nodes,
        "ranks": ordered,
    }


def collect_distributed_topology(
    rank: int,
    world_size: int,
    local_rank: int,
    *,
    expected_nodes: int,
    required_gpu_substring: str | None = None,
) -> Dict[str, Any]:
    if not dist.is_initialized():
        raise RuntimeError("Distributed topology collection requires a process group")
    local_cuda_device_count = torch.cuda.device_count()
    if not 0 <= local_rank < local_cuda_device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {local_cuda_device_count} visible GPUs"
        )
    properties = torch.cuda.get_device_properties(local_rank)
    local = {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "local_cuda_device_count": int(local_cuda_device_count),
        "slurm_node_id": os.environ.get("SLURM_NODEID"),
        "slurm_local_id": os.environ.get("SLURM_LOCALID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }
    gathered: List[Dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("Distributed topology all-gather returned an empty rank record")
    return validate_topology_records(
        [item for item in gathered if item is not None],
        world_size=world_size,
        expected_nodes=expected_nodes,
        required_gpu_substring=required_gpu_substring,
    )

