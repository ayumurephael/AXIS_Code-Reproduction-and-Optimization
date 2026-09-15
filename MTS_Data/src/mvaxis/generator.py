from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .causal_graph import CausalEdge, CausalGraphSpec, sample_doflow_graph, simulate_scm
from .data_schema import SCHEMA_VERSION, attach_channel_scales, make_channel, validate_sample
from .evidence import recognize_evidence
from .legacy_tsad_generator import generate_legacy_tsad_dataset
from .question_provider import apply_question_spec
from .question_template import sample_question_specs
from .utils import set_seed, write_jsonl


NODE_ANOMALY_TYPES = [
    "spike",
    "level_shift",
    "variance_burst",
    "oscillation",
    "stuck",
    "trend_reversal",
    "saturation",
]

EDGE_ANOMALY_TYPES = [
    "relation_break",
    "gain_change",
    "sign_flip",
    "lag_shift",
    "phase_delay",
    "parent_child_decoupling",
]

SUBGRAPH_ANOMALY_TYPES = [
    "root_phase_intervention",
    "common_cause_shock",
    "propagation_attenuation",
    "propagation_amplification",
]

ANOMALY_TYPES = NODE_ANOMALY_TYPES + EDGE_ANOMALY_TYPES + SUBGRAPH_ANOMALY_TYPES

def _global_descriptor(spec: CausalGraphSpec, channels: List[Dict[str, Any]]) -> str:
    active = [ch["channel_id"] for ch in channels if ch.get("active", True)]
    roots = [f"ch_{idx}" for idx in spec.root_nodes]
    edge_text = ", ".join(f"ch_{e.source}->ch_{e.target}(lag={e.lag})" for e in spec.edges[:8])
    if len(spec.edges) > 8:
        edge_text += ", ..."
    return (
        f"Multivariate DoFlow-style {spec.graph_type} system with {spec.active_nodes} active "
        f"channels ({', '.join(active)}). Root driver channel(s): {', '.join(roots)}. "
        f"Mechanism: {spec.mechanism}. Main directed dependencies: {edge_text}."
    )


def _channel_role(spec: CausalGraphSpec, node: int) -> str:
    if node in spec.root_nodes:
        return "cause"
    if not spec.children(node):
        return "response"
    return "sensor"


def _make_channels(spec: CausalGraphSpec) -> List[Dict[str, Any]]:
    channels: List[Dict[str, Any]] = []
    outgoing: Dict[int, List[CausalEdge]] = {i: [] for i in range(spec.max_nodes)}
    for edge in spec.edges:
        outgoing[edge.source].append(edge)
    for idx in range(spec.max_nodes):
        active = idx < spec.active_nodes
        if active:
            prior_relations = [
                {
                    "target_channel_id": f"ch_{edge.target}",
                    "relation_type": "affects",
                    "description": (
                        f"DoFlow {spec.graph_type} lag-{edge.lag} dependency "
                        f"from ch_{edge.source} to ch_{edge.target}."
                    ),
                    "expected_lag_range": [edge.lag, edge.lag],
                    "coefficient": float(edge.coefficient),
                }
                for edge in outgoing[idx]
            ]
            channel = make_channel(
                f"ch_{idx}",
                f"{spec.graph_type}_x{idx + 1}",
                _channel_role(spec, idx),
                "continuous",
                "a.u.",
                f"Active DoFlow {spec.graph_type} node X{idx + 1}.",
                prior_relations,
            )
            channel["active"] = True
            channel["graph_node"] = f"X{idx + 1}"
        else:
            channel = make_channel(
                f"ch_{idx}",
                f"padding_x{idx + 1}",
                "inactive",
                "continuous",
                "a.u.",
                "Padding channel used only to keep AXIS num_features fixed across graph families.",
                [],
            )
            channel["active"] = False
            channel["graph_node"] = None
        channels.append(channel)
    return channels


def _choose_interval(length: int) -> Tuple[int, int]:
    win = random.randint(max(16, length // 8), max(20, length // 4))
    start = random.randint(8, length - win - 8)
    return start, start + win


def _shifted(x: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return x.copy()
    return np.concatenate([np.repeat(x[0], lag), x[:-lag]])


def _add_delta_with_lag(abnormal: np.ndarray, labels: np.ndarray, start: int, end: int, channel: int, delta: np.ndarray, lag: int) -> None:
    if lag >= len(delta):
        return
    src = delta[: len(delta) - lag] if lag else delta
    dst_start = start + lag
    dst_end = min(end, dst_start + len(src))
    if dst_start >= dst_end:
        return
    abnormal[dst_start:dst_end, channel] += src[: dst_end - dst_start]
    labels[dst_start:dst_end, channel] = 1


def _propagate_delta(
    spec: CausalGraphSpec,
    abnormal: np.ndarray,
    normal: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
    seed_nodes: List[int],
    scale: float,
) -> List[int]:
    affected = set(seed_nodes)
    for node in range(spec.active_nodes):
        for edge in spec.parents(node):
            if edge.source not in affected:
                continue
            parent_delta = abnormal[start:end, edge.source] - normal[start:end, edge.source]
            if np.max(np.abs(parent_delta)) < 1e-8:
                continue
            _add_delta_with_lag(abnormal, labels, start, end, node, scale * edge.coefficient * parent_delta, edge.lag)
            affected.add(node)
    return sorted(affected)


def _node_pattern(normal: np.ndarray, start: int, end: int, channel: int, anomaly_type: str, alpha: float) -> np.ndarray:
    length = end - start
    idx = np.arange(length)
    amp = alpha * (1.5 + 0.5 * np.std(normal[:, channel]))
    base = normal[start:end, channel]
    if anomaly_type == "spike":
        return amp * np.exp(-((idx - length / 2) ** 2) / (2 * (max(2, length / 8)) ** 2))
    if anomaly_type == "level_shift":
        return np.full(length, amp)
    if anomaly_type == "variance_burst":
        return amp * np.random.randn(length)
    if anomaly_type == "oscillation":
        return amp * np.sin(np.linspace(0, 8 * np.pi, length))
    if anomaly_type == "stuck":
        return normal[start, channel] - base
    if anomaly_type == "trend_reversal":
        return base[0] - (base - base[0]) * (0.5 + alpha) - base
    if anomaly_type == "saturation":
        return np.clip(base + amp, np.percentile(normal[:, channel], 70), np.percentile(normal[:, channel], 90)) - base
    raise ValueError(f"Unsupported node anomaly type: {anomaly_type}")


def _choose_edge(spec: CausalGraphSpec) -> CausalEdge:
    return random.choice(spec.edges)


def _apply_node_anomaly(
    spec: CausalGraphSpec,
    normal: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[List[str]], List[str], List[str]]:
    abnormal = normal.copy()
    labels = np.zeros_like(normal, dtype=int)
    root = random.randrange(spec.active_nodes)
    delta = _node_pattern(normal, start, end, root, anomaly_type, alpha)
    abnormal[start:end, root] += delta
    labels[start:end, root] = 1
    affected_idx = _propagate_delta(spec, abnormal, normal, labels, start, end, [root], scale=0.45)
    causal_path = [f"ch_{node}" for node in spec.descendants([root]) if node in affected_idx]
    return abnormal, labels, affected_idx, [], [f"ch_{root}"], causal_path


def _apply_edge_anomaly(
    spec: CausalGraphSpec,
    normal: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[List[str]], List[str], List[str]]:
    abnormal = normal.copy()
    labels = np.zeros_like(normal, dtype=int)
    edge = _choose_edge(spec)
    source = edge.source
    target = edge.target
    parent = normal[start:end, source]
    child = normal[start:end, target]
    amp = alpha * (1.2 + 0.5 * np.std(child))
    if anomaly_type == "relation_break":
        abnormal[start:end, target] = child - 2.0 * edge.coefficient * parent + amp
    elif anomaly_type == "gain_change":
        abnormal[start:end, target] = child + alpha * 1.5 * edge.coefficient * parent
    elif anomaly_type == "sign_flip":
        abnormal[start:end, target] = child - 2.0 * edge.coefficient * parent
    elif anomaly_type in {"lag_shift", "phase_delay"}:
        lag = random.randint(3, 7)
        abnormal[start:end, target] = (1 - alpha) * child + alpha * _shifted(child, lag)
    elif anomaly_type == "parent_child_decoupling":
        abnormal[start:end, target] = child + amp * np.sin(np.linspace(0, 4 * np.pi, end - start) + random.random())
    else:
        raise ValueError(f"Unsupported edge anomaly type: {anomaly_type}")
    labels[start:end, target] = 1
    affected_idx = _propagate_delta(spec, abnormal, normal, labels, start, end, [target], scale=0.35)
    abnormal_edges = [[f"ch_{source}", f"ch_{target}"]]
    causal_path = [f"ch_{node}" for node in spec.descendants([target]) if node in affected_idx]
    return abnormal, labels, affected_idx, abnormal_edges, [f"ch_{target}"], causal_path


def _apply_subgraph_anomaly(
    spec: CausalGraphSpec,
    normal: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[List[str]], List[str], List[str]]:
    abnormal = normal.copy()
    labels = np.zeros_like(normal, dtype=int)
    roots = list(spec.root_nodes)
    length = end - start
    idx = np.arange(length)
    abnormal_edges: List[List[str]] = []
    if anomaly_type == "root_phase_intervention":
        for root in roots:
            driver = np.sin(2 * np.pi * (idx + spec.root_period / 2) / spec.root_period + spec.root_phase)
            abnormal[start:end, root] += alpha * 1.8 * driver
            labels[start:end, root] = 1
    elif anomaly_type == "common_cause_shock":
        shock = alpha * 1.8 * np.exp(-((idx - length / 2) ** 2) / (2 * (max(2, length / 6)) ** 2))
        for root in roots:
            abnormal[start:end, root] += shock
            labels[start:end, root] = 1
    elif anomaly_type in {"propagation_attenuation", "propagation_amplification"}:
        edge = _choose_edge(spec)
        factor = -0.8 if anomaly_type == "propagation_attenuation" else 1.2
        abnormal[start:end, edge.target] += factor * alpha * edge.coefficient * normal[start:end, edge.source]
        labels[start:end, edge.target] = 1
        roots = [edge.target]
        abnormal_edges = [[f"ch_{edge.source}", f"ch_{edge.target}"]]
    else:
        raise ValueError(f"Unsupported subgraph anomaly type: {anomaly_type}")
    affected_idx = _propagate_delta(spec, abnormal, normal, labels, start, end, roots, scale=0.50)
    causal_path = [f"ch_{node}" for node in spec.descendants(roots) if node in affected_idx]
    return abnormal, labels, affected_idx, abnormal_edges, [f"ch_{node}" for node in roots], causal_path


def _apply_anomaly(
    spec: CausalGraphSpec,
    normal: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[List[str]], List[str], List[str], str]:
    if alpha <= 0:
        return normal.copy(), np.zeros_like(normal, dtype=int), [], [], [], [], "none"
    if anomaly_type in NODE_ANOMALY_TYPES:
        abnormal, labels, affected, abnormal_edges, root_nodes, path = _apply_node_anomaly(spec, normal, start, end, anomaly_type, alpha)
        scope = "node"
    elif anomaly_type in EDGE_ANOMALY_TYPES:
        abnormal, labels, affected, abnormal_edges, root_nodes, path = _apply_edge_anomaly(spec, normal, start, end, anomaly_type, alpha)
        scope = "edge"
    else:
        abnormal, labels, affected, abnormal_edges, root_nodes, path = _apply_subgraph_anomaly(spec, normal, start, end, anomaly_type, alpha)
        scope = "subgraph"
    return abnormal, labels, affected, abnormal_edges, root_nodes, path, scope


def _target_output(
    anomaly_type: str | None,
    anomaly_scope: str,
    root_id: str | None,
    root_nodes: List[str],
    affected: List[str],
    abnormal_edges: List[List[str]],
    causal_path: List[str],
    alpha: float,
    evidence: Dict[str, Any],
    graph_type: str,
    question_type: str,
) -> Dict[str, Any]:
    flags = evidence["summary_flags"]
    is_anom = alpha > 0
    evidence_bits = []
    if flags["has_high_freq_oscillation"]:
        evidence_bits.append("high-frequency oscillation or variance burst evidence")
    if flags["has_over_fluctuation"]:
        evidence_bits.append("over-fluctuation evidence")
    if flags["has_relation_break"]:
        evidence_bits.append("parent-child relation break evidence")
    if flags["has_phase_lag"]:
        evidence_bits.append("phase-lag evidence")
    if flags["has_level_shift"]:
        evidence_bits.append("level-shift evidence")
    if flags["max_gap_level"] != "none":
        evidence_bits.append(f"{flags['max_gap_level']} left/right gap evidence")
    evidence_text = "; ".join(evidence_bits) if evidence_bits else "no strong local evidence flags"
    if is_anom:
        downstream = [ch for ch in affected if ch not in root_nodes]
        edge_text = (
            " abnormal edge(s): "
            + ", ".join(f"{src}->{dst}" for src, dst in abnormal_edges)
            + "."
            if abnormal_edges
            else ""
        )
        path_text = (
            " Propagation path: " + " -> ".join(causal_path) + "."
            if causal_path
            else " No downstream propagation path is required by the label."
        )
        if anomaly_scope == "node":
            mechanism_text = (
                f"{root_id} is the directly intervened node; "
                f"the {anomaly_type} pattern starts there before any downstream response."
            )
        elif anomaly_scope == "edge":
            mechanism_text = (
                f"the direct abnormal behavior is attached to {root_id}, the child side of the changed relation;"
                f"{edge_text}"
            )
        else:
            mechanism_text = (
                f"the intervention affects system-level propagation from {', '.join(root_nodes)};"
                f"{edge_text}"
            )
        reasoning = (
            f"The target interval is anomalous with strength alpha={alpha:.2f}. "
            f"{mechanism_text} "
            f"Ground-truth affected channels are {', '.join(affected)}. "
            f"Downstream-only channels are {', '.join(downstream) if downstream else 'none'}. "
            f"{path_text} Evidence summary: {evidence_text}."
        )
        answer = (
            f"The interval is anomalous. The direct root-cause channel is {root_id}; "
            f"the anomaly is {anomaly_scope}-level ({anomaly_type}) in the multivariate system. "
            f"Affected channels: {', '.join(affected)}. "
            f"The explanation should attribute the cause to {root_id}, not merely to downstream channels."
        )
        evidence_chain = [
            f"The target interval is labeled anomalous with strength alpha={alpha:.2f}.",
            f"The direct abnormal component is {root_id} with type {anomaly_type}.",
            f"Affected channels are {', '.join(affected) if affected else 'none'}.",
            f"Local evidence summary: {evidence_text}.",
        ]
    else:
        reasoning = (
            "The target interval is a normal counterfactual window with alpha=0.00. "
            "There are no labeled affected channels, no root-cause channel, and no abnormal edge. "
            f"Evidence summary: {evidence_text}. A conservative explanation should reject anomaly attribution."
        )
        answer = "No clear anomaly is present in the target interval."
        evidence_chain = [
            "The target interval has alpha=0.00 and no injected anomaly label.",
            "There is no labeled root-cause channel, abnormal edge, or affected channel set.",
            f"Local evidence summary: {evidence_text}.",
        ]
    del question_type
    question_answer = answer
    return {
        "fact_check": {
            "is_anomalous": is_anom,
            "root_cause_channel": root_id,
            "root_cause_channels": root_nodes,
            "affected_channels": affected,
            "anomaly_type": anomaly_type if is_anom else None,
            "anomaly_scope": anomaly_scope if is_anom else None,
            "abnormal_edges": abnormal_edges,
            "causal_path": causal_path,
            "graph_type": graph_type,
            "has_high_freq_oscillation": flags["has_high_freq_oscillation"],
            "has_over_fluctuation": flags["has_over_fluctuation"],
            "has_relation_break": flags["has_relation_break"],
            "left_right_gap_level": flags["max_gap_level"],
            "state_response_consistency": "inconsistent" if flags["has_relation_break"] else "consistent",
            "evidence_used": ["target_interval", "evidence_card", "channel_metadata", "axis_embedding_hints"],
        },
        "evidence_chain": evidence_chain,
        "reasoning_summary": reasoning,
        "question_answer": question_answer,
        "final_answer": answer,
        "abnormality_score": float(alpha),
        "answer_confidence": float(0.55 + 0.4 * abs(alpha - 0.3)) if is_anom else 0.85,
    }


def _axis_window(
    sample: Dict[str, Any],
    question_type: str,
    alpha: float,
) -> Dict[str, Any]:
    start = int(sample["target_interval"]["start"])
    end = int(sample["target_interval"]["end"])
    label = dict(sample["synthetic_label"])
    target = sample["target_output"]
    has_anomaly = bool(alpha > 0)
    if has_anomaly:
        description = (
            f"{label.get('anomaly_scope')} anomaly {label.get('anomaly_type')} "
            f"with root {sample['root_cause'].get('channel_id')} and affected channels "
            f"{', '.join(label.get('affected_channels') or [])}."
        )
    else:
        description = "Normal counterfactual window without injected anomaly."
    return {
        "window_range": [start, end],
        "question": sample["question"],
        "answer": target.get("final_answer", ""),
        "question_type": question_type,
        "has_anomaly": has_anomaly,
        "anomaly_descriptions": [description] if description else [],
        "mv_label": {
            "root_cause_channel": sample["root_cause"].get("channel_id"),
            "root_cause_channels": label.get("root_cause_channels", []),
            "affected_channels": label.get("affected_channels", []),
            "anomaly_type": label.get("anomaly_type"),
            "anomaly_scope": label.get("anomaly_scope"),
            "abnormal_edges": label.get("abnormal_edges", []),
            "causal_path": label.get("causal_path", []),
            "graph_type": label.get("graph_type"),
            "counterfactual_normal_sample_id": label.get("counterfactual_normal_sample_id"),
        },
    }


def make_sample(
    base_id: int,
    variant_idx: int,
    normal: np.ndarray,
    channels: List[Dict[str, Any]],
    spec: CausalGraphSpec,
    alpha: float,
    data_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data_cfg = data_cfg or {}
    start, end = _choose_interval(normal.shape[0])
    anomaly_pool = list(data_cfg.get("anomaly_types") or ANOMALY_TYPES)
    anomaly_type = random.choice(anomaly_pool)
    question_spec = sample_question_specs(1, random.randint(0, 10_000_000))[0]
    question_type = question_spec.question_id
    effective_alpha = float(alpha) * float(data_cfg.get("anomaly_strength_multiplier", 1.0))
    values, labels, affected_idx, abnormal_edges, root_nodes, causal_path, anomaly_scope = _apply_anomaly(
        spec,
        normal,
        start,
        end,
        anomaly_type,
        effective_alpha,
    )
    affected = [f"ch_{c}" for c in affected_idx]
    root_id = root_nodes[0] if alpha > 0 and root_nodes else None
    sample = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"base_{base_id:05d}_alpha_{variant_idx:02d}",
        "base_sample_id": f"base_{base_id:05d}",
        "source": "synthetic_doflow_scm",
        "original_data": {
            "time_series": values.round(6).tolist(),
            "normal_series": normal.round(6).tolist(),
            "global_descriptor": _global_descriptor(spec, channels),
        },
        "normal_series": normal.round(6).tolist(),
        "series": {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "values": values.round(6).tolist(),
            "labels": labels.astype(int).tolist(),
            "timestamps": None,
        },
        "channels": [dict(ch) for ch in channels],
        "causal_graph": spec.to_dict(),
        "target_interval": {"start": start, "end": end},
        "question_type": question_type,
        "question": question_spec.text,
        "root_cause": {
            "channel_id": root_id,
            "time_index": start if alpha > 0 else None,
            "interval": [start, end] if alpha > 0 else None,
            "provided_by": "synthetic_label",
        },
        "synthetic_label": {
            "is_synthetic": True,
            "base_sample_id": f"base_{base_id:05d}",
            "abnormality_strength_alpha": float(alpha),
            "effective_anomaly_strength_alpha": float(effective_alpha),
            "anomaly_type": anomaly_type if alpha > 0 else None,
            "anomaly_scope": anomaly_scope if alpha > 0 else None,
            "graph_type": spec.graph_type,
            "mechanism": spec.mechanism,
            "affected_channels": affected,
            "abnormal_edges": abnormal_edges,
            "root_cause_channels": root_nodes,
            "causal_path": causal_path,
            "counterfactual_normal_sample_id": f"base_{base_id:05d}_alpha_00",
            "generator_profile": data_cfg.get("profile", "default"),
        },
        "evidence_card": {},
        "target_output": {},
    }
    attach_channel_scales(sample)
    sample["evidence_card"] = recognize_evidence(sample)
    sample["target_output"] = _target_output(
        anomaly_type if alpha > 0 else None,
        anomaly_scope,
        root_id,
        root_nodes,
        affected,
        abnormal_edges,
        causal_path,
        min(1.0, effective_alpha),
        sample["evidence_card"],
        spec.graph_type,
        question_type,
    )
    sample["windows"] = [_axis_window(sample, question_type, alpha)]
    sample = apply_question_spec(sample, question_spec)
    validate_sample(sample)
    return sample


def _generate_doflow_dataset(config: Dict[str, Any]) -> Dict[str, str]:
    set_seed(int(config.get("seed", 72)))
    data_cfg = config["data"]
    out_dir = Path(data_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = list(data_cfg["alpha_values"])
    base_ids = list(range(int(data_cfg["num_base_systems"])))
    random.shuffle(base_ids)
    n_train = int(len(base_ids) * float(data_cfg["train_ratio"]))
    n_val = int(len(base_ids) * float(data_cfg["val_ratio"]))
    split_map = {}
    for i, bid in enumerate(base_ids):
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        split_map[bid] = split
    rows = {"train": [], "val": [], "test": []}
    for bid in range(int(data_cfg["num_base_systems"])):
        spec = sample_doflow_graph(data_cfg)
        channels = _make_channels(spec)
        normal = simulate_scm(spec, int(data_cfg["length"]), float(data_cfg["noise_scale"]))
        for j, alpha in enumerate(alpha_values):
            rows[split_map[bid]].append(make_sample(bid, j, normal, channels, spec, float(alpha), data_cfg))
    paths: Dict[str, str] = {}
    for split, split_rows in rows.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(split_rows, path)
        paths[split] = str(path)
    return paths


def generate_dataset(config: Dict[str, Any]) -> Dict[str, str]:
    """Generate the current mvaxis dataset schema.

    The default backend is the original TSAD_dataset_gen-axis multivariate
    generator, converted through the mvaxis schema adapter. Set
    data.generator_backend="doflow" to use the previous four-graph DoFlow
    sampler for backwards-compatible experiments.
    """
    backend = str(config.get("data", {}).get("generator_backend", "legacy_tsad")).lower()
    if backend in {"doflow", "synthetic_doflow_scm"}:
        return _generate_doflow_dataset(config)
    if backend in {"legacy_tsad", "legacy_axis", "tsad_dataset_gen_axis"}:
        return generate_legacy_tsad_dataset(config)
    raise ValueError(f"Unknown data.generator_backend: {backend}")
