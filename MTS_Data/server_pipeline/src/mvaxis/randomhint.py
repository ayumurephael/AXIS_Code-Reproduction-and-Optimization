from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict

import numpy as np


def _seed_from_context(base_seed: int, context: str) -> int:
    payload = f"{int(base_seed)}|{context}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2 ** 32)


def _sample_unit_vector(dim: int, *, seed: int) -> np.ndarray:
    if int(dim) <= 0:
        raise ValueError(f"Random hint dimension must be positive, got {dim}.")
    rng = np.random.default_rng(int(seed))
    vector = rng.standard_normal(int(dim), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        vector = np.zeros((int(dim),), dtype=np.float32)
        vector[0] = 1.0
        return vector
    return vector / norm


def _randomize_branch(
    branch_name: str,
    tensor_like: Any,
    *,
    base_seed: int,
    context: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    tensor = np.asarray(tensor_like, dtype=np.float32)
    if tensor.ndim != 2:
        raise ValueError(
            f"Random hint branch '{branch_name}' must have shape [num_tokens, dim], got {list(tensor.shape)}."
        )
    num_tokens, dim = int(tensor.shape[0]), int(tensor.shape[1])
    if num_tokens <= 0:
        return tensor.copy(), {
            "num_tokens": num_tokens,
            "dim": dim,
            "distribution": "unit_sphere",
            "shared_vector_per_token_family": True,
        }
    seed = _seed_from_context(base_seed, f"{context}|{branch_name}|{num_tokens}|{dim}")
    unit_vector = _sample_unit_vector(dim, seed=seed)
    randomized = np.repeat(unit_vector[None, :], num_tokens, axis=0)
    return randomized.astype(np.float32, copy=False), {
        "num_tokens": num_tokens,
        "dim": dim,
        "distribution": "unit_sphere",
        "shared_vector_per_token_family": True,
        "seed": int(seed),
    }


def build_random_embedding_hints(
    hint_bundle: Dict[str, Any],
    *,
    base_seed: int = 72,
    context: str = "",
) -> Dict[str, Any]:
    embeddings = dict((hint_bundle or {}).get("embeddings") or {})
    if not embeddings:
        raise ValueError("Cannot build random hints from an empty hint bundle.")
    randomized_embeddings: Dict[str, np.ndarray] = {}
    random_metadata: Dict[str, Any] = {}
    for branch_name in ("global", "channel"):
        tensor_like = embeddings.get(branch_name)
        if tensor_like is None:
            continue
        randomized, metadata = _randomize_branch(
            branch_name,
            tensor_like,
            base_seed=int(base_seed),
            context=str(context),
        )
        randomized_embeddings[branch_name] = randomized
        random_metadata[branch_name] = metadata
    if not randomized_embeddings:
        raise ValueError("Random hint generation requires at least one global/channel tensor.")
    trace = copy.deepcopy((hint_bundle or {}).get("trace") or {})
    trace["random_hint"] = {
        "enabled": True,
        "base_seed": int(base_seed),
        "context": str(context),
        "branches": random_metadata,
    }
    return {
        "embeddings": randomized_embeddings,
        "trace": trace,
        "random_metadata": random_metadata,
    }
