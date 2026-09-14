from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CausalEdge:
    source: int
    target: int
    coefficient: float
    lag: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": f"ch_{self.source}",
            "target": f"ch_{self.target}",
            "coefficient": float(self.coefficient),
            "lag": int(self.lag),
            "relation_type": "affects",
        }


@dataclass
class CausalGraphSpec:
    graph_type: str
    mechanism: str
    active_nodes: int
    max_nodes: int
    canonical_nodes: int
    root_nodes: List[int]
    beta: List[float]
    edges: List[CausalEdge]
    root_amplitude: float
    root_period: float
    root_phase: float
    gamma: float = 0.3
    noise_divisor: float = 1.0
    root_driver_type: str = "sin"
    independent_noise_scale: float = 0.0

    def parents(self, node: int) -> List[CausalEdge]:
        return [edge for edge in self.edges if edge.target == node]

    def children(self, node: int) -> List[int]:
        return [edge.target for edge in self.edges if edge.source == node]

    def descendants(self, roots: Iterable[int]) -> List[int]:
        seen = set(roots)
        frontier = list(roots)
        while frontier:
            node = frontier.pop(0)
            for child in self.children(node):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return sorted(seen)

    def active_channel_ids(self) -> List[str]:
        return [f"ch_{i}" for i in range(self.active_nodes)]

    def edge_between(self, source: int, target: int) -> CausalEdge | None:
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_type": self.graph_type,
            "mechanism": self.mechanism,
            "active_nodes": int(self.active_nodes),
            "max_nodes": int(self.max_nodes),
            "canonical_nodes": int(self.canonical_nodes),
            "root_nodes": [f"ch_{i}" for i in self.root_nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "structural_equation": {
                "self_lag_beta": {f"ch_{i}": float(self.beta[i]) for i in range(self.active_nodes)},
                "root_driver": {
                    "type": str(self.root_driver_type),
                    "amplitude": float(self.root_amplitude),
                    "period": float(self.root_period),
                    "phase": float(self.root_phase),
                },
                "gamma": float(self.gamma),
                "noise_divisor": float(self.noise_divisor),
                "independent_noise_scale": float(self.independent_noise_scale),
            },
            "appendix_reference": "DoFlow Appendix D: Tree, Diamond, FC-Layer, Chain SCM families.",
        }


def _jitter(value: float, amount: float) -> float:
    if amount <= 0:
        return value
    return value + random.uniform(-amount, amount)


def _edges(raw: Sequence[Tuple[int, int, float]], jitter: float) -> List[CausalEdge]:
    return [CausalEdge(src - 1, dst - 1, _jitter(coeff, jitter)) for src, dst, coeff in raw]


def _tree(max_nodes: int, mechanism: str, jitter: float) -> CausalGraphSpec:
    active = 8
    if max_nodes < active:
        raise ValueError("DoFlow Tree requires at least 8 configured channels")
    return CausalGraphSpec(
        graph_type="tree",
        mechanism=mechanism,
        active_nodes=active,
        max_nodes=max_nodes,
        canonical_nodes=8,
        root_nodes=[0],
        beta=[0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2],
        edges=_edges(
            [
                (1, 2, 0.3),
                (1, 3, -0.3),
                (2, 4, 0.3),
                (2, 5, -0.3),
                (3, 6, 0.5),
                (3, 7, -0.5),
                (7, 8, 0.5),
            ],
            jitter,
        ),
        root_amplitude=1.0,
        root_period=20.0,
        root_phase=2 * math.pi / 9,
        noise_divisor=4.0,
    )


def _diamond(max_nodes: int, mechanism: str, jitter: float) -> CausalGraphSpec:
    active = 10
    if max_nodes < active:
        raise ValueError("DoFlow Diamond requires at least 10 configured channels")
    return CausalGraphSpec(
        graph_type="diamond",
        mechanism=mechanism,
        active_nodes=active,
        max_nodes=max_nodes,
        canonical_nodes=10,
        root_nodes=[0],
        beta=[0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
        edges=_edges(
            [
                (1, 2, 0.3),
                (1, 3, 0.3),
                (2, 4, 0.3),
                (2, 6, 0.3),
                (3, 5, 0.5),
                (3, 7, 0.5),
                (4, 8, 0.5),
                (6, 8, 0.5),
                (5, 9, 0.5),
                (7, 9, 0.5),
                (8, 10, 0.5),
                (9, 10, 0.5),
            ],
            jitter,
        ),
        root_amplitude=1.0,
        root_period=20.0,
        root_phase=2 * math.pi / 9,
    )


def _fc_layer(max_nodes: int, mechanism: str, jitter: float) -> CausalGraphSpec:
    active = 10
    if max_nodes < active:
        raise ValueError("DoFlow FC-Layer requires at least 10 configured channels")
    raw: List[Tuple[int, int, float]] = []
    for hidden in range(4, 8):
        for input_node in range(1, 4):
            raw.append((input_node, hidden, 0.2 + 0.1 * (input_node - 1)))
    for output in range(8, 11):
        for hidden in range(4, 8):
            raw.append((hidden, output, 0.2))
    return CausalGraphSpec(
        graph_type="fc_layer",
        mechanism=mechanism,
        active_nodes=active,
        max_nodes=max_nodes,
        canonical_nodes=10,
        root_nodes=[0, 1, 2],
        beta=[0.5, 0.6, 0.7, 0.4, 0.45, 0.5, 0.55, 0.3, 0.35, 0.4],
        edges=_edges(raw, jitter),
        root_amplitude=1.5,
        root_period=15.0,
        root_phase=4 * math.pi / 9,
        gamma=0.30,
    )


def _chain(max_nodes: int, mechanism: str, jitter: float, chain_nodes: int) -> CausalGraphSpec:
    active = min(chain_nodes, max_nodes)
    if active < 2:
        raise ValueError("DoFlow Chain requires at least 2 configured channels")
    raw: List[Tuple[int, int, float]] = []
    for node in range(2, active + 1):
        coeff = 0.2
        if node == 2:
            coeff += 0.7
            raw.append((1, node, coeff))
        else:
            raw.append((node - 1, node, 0.2))
            raw.append((1, node, 0.7))
    return CausalGraphSpec(
        graph_type="chain",
        mechanism=mechanism,
        active_nodes=active,
        max_nodes=max_nodes,
        canonical_nodes=50,
        root_nodes=[0],
        beta=[0.6 for _ in range(active)],
        edges=_edges(raw, jitter),
        root_amplitude=1.0,
        root_period=20.0,
        root_phase=2 * math.pi / 9,
    )


def sample_doflow_graph(config: Dict[str, Any]) -> CausalGraphSpec:
    max_nodes = int(config.get("num_channels", config.get("max_num_channels", 10)))
    graph_cfg = config.get("causal_graph", {})
    graph_types = list(graph_cfg.get("graph_types", ["tree", "diamond", "fc_layer", "chain"]))
    mechanism_types = list(graph_cfg.get("mechanisms", ["additive", "nonlinear"]))
    graph_type = str(random.choice(graph_types)).lower().replace("-", "_")
    mechanism = str(random.choice(mechanism_types)).lower()
    if mechanism not in {"additive", "nonlinear", "exponential", "tanh", "mixed"}:
        raise ValueError(f"Unsupported SCM mechanism: {mechanism}")
    jitter = float(graph_cfg.get("coefficient_jitter", 0.0))
    if graph_type == "tree":
        spec = _tree(max_nodes, mechanism, jitter)
    elif graph_type == "diamond":
        spec = _diamond(max_nodes, mechanism, jitter)
    elif graph_type in {"fc_layer", "fully_connected_layer", "fclayer"}:
        spec = _fc_layer(max_nodes, mechanism, jitter)
    elif graph_type == "chain":
        chain_nodes = int(graph_cfg.get("chain_num_channels", max_nodes))
        spec = _chain(max_nodes, mechanism, jitter, chain_nodes)
    else:
        raise ValueError(f"Unsupported DoFlow graph type: {graph_type}")
    coeff_scale = float(graph_cfg.get("coefficient_scale", 1.0))
    if coeff_scale != 1.0:
        spec.edges = [
            CausalEdge(edge.source, edge.target, edge.coefficient * coeff_scale, edge.lag)
            for edge in spec.edges
        ]
    lag_choices = graph_cfg.get("lag_choices")
    if lag_choices:
        choices = [int(x) for x in lag_choices]
        spec.edges = [
            CausalEdge(edge.source, edge.target, edge.coefficient, int(random.choice(choices)))
            for edge in spec.edges
        ]
    beta_range = graph_cfg.get("beta_range")
    if beta_range and len(beta_range) == 2:
        lo, hi = float(beta_range[0]), float(beta_range[1])
        spec.beta = [random.uniform(lo, hi) for _ in spec.beta]
    amp_range = graph_cfg.get("root_amplitude_range")
    if amp_range and len(amp_range) == 2:
        spec.root_amplitude = random.uniform(float(amp_range[0]), float(amp_range[1]))
    period_range = graph_cfg.get("root_period_range")
    if period_range and len(period_range) == 2:
        spec.root_period = random.uniform(float(period_range[0]), float(period_range[1]))
    if graph_cfg.get("random_root_phase", False):
        spec.root_phase = random.uniform(0.0, 2 * math.pi)
    driver_types = graph_cfg.get("root_driver_types")
    if driver_types:
        spec.root_driver_type = str(random.choice(list(driver_types))).lower()
    spec.noise_divisor = float(graph_cfg.get("noise_divisor", spec.noise_divisor))
    spec.independent_noise_scale = float(graph_cfg.get("independent_noise_scale", 0.0))
    return spec


def simulate_scm(spec: CausalGraphSpec, length: int, noise_scale: float) -> np.ndarray:
    values = np.zeros((length, spec.max_nodes), dtype=float)
    active = spec.active_nodes
    beta = np.asarray(spec.beta, dtype=float)
    phase_jitter = np.linspace(0.0, math.pi / 3, max(1, len(spec.root_nodes)))
    for t in range(1, length):
        noise = np.random.normal(0.0, noise_scale, size=active)
        local_noise = np.random.normal(0.0, spec.independent_noise_scale, size=active)
        for root_pos, node in enumerate(spec.root_nodes):
            if spec.root_driver_type == "sin":
                driver = spec.root_amplitude * math.sin(
                    2 * math.pi * t / spec.root_period + spec.root_phase + phase_jitter[root_pos]
                )
            elif spec.root_driver_type == "random_walk":
                driver = 0.18 * values[t - 1, node] + spec.root_amplitude * noise[node]
            elif spec.root_driver_type == "pulse":
                driver = spec.root_amplitude * (1.0 if random.random() < 0.035 else 0.0) * random.choice([-1.0, 1.0])
            elif spec.root_driver_type == "ar_noise":
                driver = spec.root_amplitude * noise[node]
            else:
                driver = spec.root_amplitude * noise[node]
            values[t, node] = beta[node] * values[t - 1, node] + driver + noise[node] + local_noise[node]
        for node in range(active):
            if node in spec.root_nodes:
                continue
            parent_sum = sum(edge.coefficient * values[t - edge.lag, edge.source] for edge in spec.parents(node))
            u = noise[node]
            prev = values[t - 1, node]
            if spec.mechanism == "additive":
                values[t, node] = beta[node] * prev + parent_sum + u / spec.noise_divisor + local_noise[node]
            elif spec.mechanism == "exponential":
                clipped_parent = float(np.clip(parent_sum, -2.5, 2.5))
                values[t, node] = (
                    beta[node] * prev
                    + 0.38 * (math.exp(0.65 * clipped_parent) - 1.0)
                    + 0.22 * math.tanh(prev * clipped_parent)
                    + u / max(1e-6, spec.noise_divisor)
                    + local_noise[node]
                )
            elif spec.mechanism == "tanh":
                clipped_parent = float(np.clip(parent_sum, -3.0, 3.0))
                values[t, node] = (
                    beta[node] * prev
                    + 1.25 * math.tanh(clipped_parent)
                    + 0.14 * np.sign(clipped_parent) * clipped_parent * clipped_parent
                    + u / max(1e-6, spec.noise_divisor)
                    + local_noise[node]
                )
            elif spec.mechanism == "mixed":
                clipped_parent = float(np.clip(parent_sum, -2.5, 2.5))
                if node % 3 == 0:
                    relation_term = 0.55 * (math.exp(0.55 * clipped_parent) - 1.0)
                elif node % 3 == 1:
                    relation_term = 1.15 * math.tanh(clipped_parent)
                else:
                    relation_term = parent_sum + 0.25 * math.sin(clipped_parent + prev)
                values[t, node] = beta[node] * prev + relation_term + u / max(1e-6, spec.noise_divisor) + local_noise[node]
            elif spec.graph_type == "tree":
                values[t, node] = beta[node] * prev * (abs(u) + 0.5) + parent_sum + local_noise[node]
            elif spec.graph_type == "diamond":
                values[t, node] = 0.5 * math.exp(np.clip(beta[node] * prev, -2.0, 2.0)) + abs(u) + parent_sum + local_noise[node]
            elif spec.graph_type == "fc_layer":
                clipped_parent = float(np.clip(parent_sum, -2.0, 2.0))
                values[t, node] = (
                    beta[node] * prev
                    + parent_sum
                    + spec.gamma * (math.exp(clipped_parent - 1.0) - 1.0)
                    - spec.gamma * math.tanh(clipped_parent)
                    + u
                    + local_noise[node]
                )
            else:
                pred_edge = spec.edge_between(node - 1, node) if node > 0 else None
                root_edge = spec.edge_between(0, node)
                pred_term = (pred_edge.coefficient if pred_edge else 0.0) * values[t - 1, node - 1] if node > 0 else 0.0
                root_term = (root_edge.coefficient if root_edge else 0.0) * values[t - 1, 0]
                values[t, node] = (beta[node] * prev + 0.5) * abs(u) + pred_term + root_term + local_noise[node]
        values[t, :active] = np.clip(values[t, :active], -8.0, 8.0)
    for node in range(active):
        x = values[:, node]
        values[:, node] = (x - x.mean()) / (x.std() + 1e-8)
    return values
