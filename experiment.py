import json
import math
import argparse
import random
import re
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    ExperimentConfig,
    FUNCTIONAL_ABLATION_CHOICES,
    MEASURED_SEARCH_VERIFY_COST_RATIO,
    MUTATION_END_START_RATIO,
    SEARCH_VERIFY_RATIO_SOURCE,
    build_default_config,
    configure_ablation_strategy_space,
    get_functional_ablation_spec,
)
from .core.evolution import EvolutionEnvironment
from .llm import apply_model_overrides, resolve_model
from .core.prompts import normalize_graph_edges
from .utils.answers import (
    choice_letters_from_choices,
    evaluate_final_agent_answers,
    parse_mmlupro_choice_answer,
)
from .two_window import (
    add_two_window_fault_args,
    build_controller as build_two_window_fault_controller,
    protocol_enabled as two_window_fault_enabled,
    validate_fixed_horizon as validate_two_window_fixed_horizon,
    validate_protocol_inputs as validate_two_window_protocol_inputs,
)


def _safe_model_dirname(model_name: str) -> str:
    name = str(model_name or "unknown_model").strip()
    if not name:
        name = "unknown_model"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _path_with_model_folder(path: Path | str, model_name: str) -> Path:
    base = Path(path)
    model_dir = _safe_model_dirname(model_name)
    if base.parent.name == model_dir:
        return base
    return base.parent / model_dir / base.name


def load_real_graph(jsonl_path: str, line_idx: int = 0):
    try:
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i == line_idx:
                    graph_dict = json.loads(line)
                    normalize_graph_edges(graph_dict)
                    return graph_dict
        raise ValueError(f"File {jsonl_path} does not contain line {line_idx}.")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Input graph not found: {jsonl_path}. Generate a dataset or pass --input-graph."
        ) from exc


def _build_agent_series(agent_history, key):
    if not agent_history:
        return {}
    agent_ids = [
        agent["agent_id"]
        for agent in agent_history[0].get("agents", [])
    ]
    series = {agent_id: [] for agent_id in agent_ids}
    for generation_snapshot in agent_history:
        snapshot_by_id = {agent["agent_id"]: agent for agent in generation_snapshot.get("agents", [])}
        for agent_id in agent_ids:
            series[agent_id].append(snapshot_by_id.get(agent_id, {}).get(key))
    return series


def plot_evolution_metrics(history_stats, agent_history=None, config: ExperimentConfig | None = None, output_path: Path | None = None):
    """Plot averaged curves together with per-agent overlays for the main result figure."""
    if not history_stats:
        print("[Plot] No history data available.")
        return

    generations = [stat["gen"] for stat in history_stats]
    avg_fitness = [stat["avg_fitness"] for stat in history_stats]
    avg_rho = [stat.get("avg_rho", stat["avg_h_final"]) for stat in history_stats]
    self_ratio = [stat["self_reflection_ratio"] for stat in history_stats]
    avg_edge_risk = [stat.get("avg_edge_risk", 0.0) for stat in history_stats]
    agent_fitness_series = _build_agent_series(agent_history, "fitness")
    agent_h_series = _build_agent_series(agent_history, "h_final")
    agent_a_self_series = _build_agent_series(agent_history, "a_self")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for agent_id, values in agent_fitness_series.items():
        axes[0].plot(generations, values, color="#a9bcd0", linewidth=1.0, alpha=0.55)
    axes[0].plot(generations, avg_fitness, color="#1d3557", marker="o", linewidth=2.5, markersize=5, label="Average")
    axes[0].set_title("Average Fitness", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Generation", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].legend()

    for agent_id, values in agent_h_series.items():
        axes[1].plot(generations, values, color="#efb0b8", linewidth=1.0, alpha=0.55)
    axes[1].plot(generations, avg_rho, color="#b23a48", marker="s", linewidth=2.5, markersize=5, label="Average")
    axes[1].set_title("Propagation Risk Rho", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Generation", fontsize=12)
    axes[1].set_ylabel("Probability", fontsize=12)
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[1].legend()

    for agent_id, values in agent_a_self_series.items():
        axes[2].plot(generations, values, linewidth=1.0, alpha=0.55)
    axes[2].plot(generations, self_ratio, color="#2a9d8f", marker="^", linewidth=2.5, markersize=5, label="Average")
    axes[2].set_title("Self-Reflection Ratio", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Generation", fontsize=12)
    axes[2].set_ylabel("Ratio", fontsize=12)
    axes[2].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()

    axes[3].plot(generations, avg_edge_risk, color="#6a4c93", marker="d", linewidth=2.5, markersize=5, label="Average")
    axes[3].set_title("Average Edge Risk", fontsize=14, fontweight="bold")
    axes[3].set_xlabel("Generation", fontsize=12)
    axes[3].set_ylabel("Risk", fontsize=12)
    axes[3].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[3].legend()

    plt.tight_layout()
    config = config or build_default_config()
    output_path = output_path or _path_with_model_folder(config.paths.evolution_figure_path, config.llm.generation_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved figure to {output_path}")
    plt.close(fig)


def _format_metric(value):
    return "N/A" if value is None else f"{value:.3f}"


def _build_undirected_adj(graph_dict):
    num_nodes = len(graph_dict.get("nodes", []))
    adj = [set() for _ in range(num_nodes)]
    for edge in graph_dict.get("edges", []):
        if len(edge) < 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if src == dst or src >= num_nodes or dst >= num_nodes or src < 0 or dst < 0:
            continue
        adj[src].add(dst)
        adj[dst].add(src)
    return adj


def _connected_components(adj):
    visited = set()
    components = []
    for node_id in range(len(adj)):
        if node_id in visited:
            continue
        stack = [node_id]
        component = []
        visited.add(node_id)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _avg_shortest_path_lcc(adj):
    components = _connected_components(adj)
    if not components:
        return 0.0, 0.0

    largest_component = max(components, key=len)
    lcc_ratio = len(largest_component) / len(adj) if adj else 0.0

    if len(largest_component) <= 1:
        return 0.0, lcc_ratio

    total_distance = 0
    pair_count = 0
    lcc_set = set(largest_component)
    for source in largest_component:
        queue = [(source, 0)]
        visited = {source}
        head = 0
        while head < len(queue):
            node, dist = queue[head]
            head += 1
            for neighbor in adj[node]:
                if neighbor in lcc_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    total_distance += dist + 1
                    pair_count += 1

    avg_path = total_distance / pair_count if pair_count else 0.0
    return avg_path, lcc_ratio


def _avg_clustering(adj):
    if not adj:
        return 0.0

    clustering_values = []
    for node_id, neighbors in enumerate(adj):
        degree = len(neighbors)
        if degree < 2:
            clustering_values.append(0.0)
            continue

        links_between_neighbors = 0
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adj[neighbor_list[i]]:
                    links_between_neighbors += 1

        possible_links = degree * (degree - 1) / 2
        clustering_values.append(links_between_neighbors / possible_links if possible_links else 0.0)

    return sum(clustering_values) / len(clustering_values)


def _spectral_gap(adj):
    num_nodes = len(adj)
    if num_nodes <= 1:
        return 0.0

    if torch is not None:
        laplacian = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
        for i in range(num_nodes):
            laplacian[i, i] = len(adj[i])
            for j in adj[i]:
                laplacian[i, j] = -1.0

        eigenvalues = torch.linalg.eigvalsh(laplacian)
        sorted_eigs = torch.sort(eigenvalues).values
        return float(max(0.0, sorted_eigs[1].item()))

    laplacian = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        laplacian[i, i] = len(adj[i])
        for j in adj[i]:
            laplacian[i, j] = -1.0

    eigenvalues = np.linalg.eigvalsh(laplacian)
    sorted_eigs = np.sort(eigenvalues)
    return float(max(0.0, sorted_eigs[1]))


def _degree_heterogeneity(adj):
    degrees = [len(neighbors) for neighbors in adj]
    if not degrees:
        return 0.0
    mean_degree = sum(degrees) / len(degrees)
    if mean_degree <= 1e-12:
        return 0.0
    variance = sum((degree - mean_degree) ** 2 for degree in degrees) / len(degrees)
    return math.sqrt(variance) / mean_degree


def compute_topology_metrics(graph_dict):
    adj = _build_undirected_adj(graph_dict)
    edge_count_undirected = sum(len(neighbors) for neighbors in adj) / 2 if adj else 0.0
    spectral_gap = _spectral_gap(adj)
    avg_path_length, lcc_ratio = _avg_shortest_path_lcc(adj)
    avg_clustering = _avg_clustering(adj)
    degree_heterogeneity = _degree_heterogeneity(adj)
    edge_normalized_spectral_gap = spectral_gap / edge_count_undirected if edge_count_undirected > 0 else 0.0

    return {
        "spectral_gap": spectral_gap,
        "avg_path_length": avg_path_length,
        "avg_clustering": avg_clustering,
        "edge_normalized_spectral_gap": edge_normalized_spectral_gap,
        "spectral_efficiency": edge_normalized_spectral_gap,
        "topo_efficiency_legacy": edge_normalized_spectral_gap,
        "lcc_ratio": lcc_ratio,
        "degree_heterogeneity": degree_heterogeneity,
    }


def safe_edge_t(transmission_matrix: list, src: int, dst: int, default_if_missing: float = 1.0) -> float:
    try:
        value = transmission_matrix[dst][src]
    except Exception:
        return float(default_if_missing)
    if value is None:
        return float(default_if_missing)
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return float(default_if_missing)


def compute_topology_risk_metrics(
    graph_dict,
    agents,
    transmission_matrix=None,
    beta: float = 1.0,
    source_pi_by_edge=None,
):
    edges = graph_dict.get("edges", [])
    if not edges:
        return {
            "avg_edge_risk": 0.0,
            "total_edge_risk": 0.0,
            "avg_adjusted_edge_risk": 0.0,
            "total_adjusted_edge_risk": 0.0,
            "high_risk_edge_ratio": 0.0,
        }

    source_risks = []
    adjusted_source_risks = []
    high_risk_count = 0
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        src = int(edge[0])
        dst = int(edge[1])
        if src not in agents:
            continue
        current_pi_j = float(getattr(agents[src], "predicted_prob", getattr(agents[src], "pi", 0.0)) or 0.0)
        pi_j = float((source_pi_by_edge or {}).get((src, dst), current_pi_j) or 0.0)
        rho_i = float(getattr(agents.get(dst), "rho", 0.0) or 0.0) if dst in agents else 0.0
        edge_t = edge_transmission(transmission_matrix or [], src, dst)
        edge_risk = edge_t * pi_j
        adjusted_edge_risk = edge_risk * (1.0 + rho_i)
        edge_delta = (1.0 - pi_j) - float(beta) * adjusted_edge_risk
        source_risks.append(edge_risk)
        adjusted_source_risks.append(adjusted_edge_risk)
        if edge_delta < 0.0:
            high_risk_count += 1

    if not source_risks:
        return {
            "avg_edge_risk": 0.0,
            "total_edge_risk": 0.0,
            "avg_adjusted_edge_risk": 0.0,
            "total_adjusted_edge_risk": 0.0,
            "high_risk_edge_ratio": 0.0,
        }

    total_edge_risk = sum(source_risks)
    return {
        "avg_edge_risk": total_edge_risk / len(source_risks),
        "total_edge_risk": total_edge_risk,
        "avg_adjusted_edge_risk": sum(adjusted_source_risks) / len(adjusted_source_risks),
        "total_adjusted_edge_risk": sum(adjusted_source_risks),
        "high_risk_edge_ratio": high_risk_count / len(source_risks),
    }


def source_pi_by_edge_from_risk_state(risk_state: Any) -> dict[tuple[int, int], float]:
    if risk_state is None:
        return {}
    per_edge = dict((getattr(risk_state, "details", {}) or {}).get("per_edge", {}) or {})
    result: dict[tuple[int, int], float] = {}
    for edge_key, edge_detail in per_edge.items():
        try:
            src_text, dst_text = str(edge_key).split("->", 1)
            result[(int(src_text), int(dst_text))] = float(
                (edge_detail or {}).get("source_message_pi", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            continue
    return result


def compute_transmission_metrics(edge_transmissions):
    if not edge_transmissions:
        return {
            "avg_edge_transmission": 0.0,
            "max_edge_transmission": 0.0,
            "observed_transmission_coverage": 0.0,
            "unknown_transmission_edge_count": 0,
        }

    observed_rows = [entry for entry in edge_transmissions if bool(entry.get("transmission_observed", True))]
    transmission_values = [float(entry.get("transmission_strength", 0.0) or 0.0) for entry in observed_rows]
    return {
        "avg_edge_transmission": (
            sum(transmission_values) / len(transmission_values) if transmission_values else 0.0
        ),
        "max_edge_transmission": max(transmission_values) if transmission_values else 0.0,
        "observed_transmission_coverage": len(observed_rows) / len(edge_transmissions),
        "unknown_transmission_edge_count": len(edge_transmissions) - len(observed_rows),
    }


def normalize_edge_transmission_records(edge_transmissions):
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value or 0.0)))

    def infer_rho_from_h_pi(h: float, pi: float) -> float:
        h = _clamp01(h)
        pi = _clamp01(pi)
        if pi >= 1.0:
            return 0.0
        return _clamp01((h - pi) / max(1e-12, 1.0 - pi))

    def infer_pi_from_h_rho(h: float, rho: float) -> float:
        h = _clamp01(h)
        rho = _clamp01(rho)
        if rho >= 1.0:
            return 0.0
        return _clamp01((h - rho) / max(1e-12, 1.0 - rho))

    normalized = []
    for edge in edge_transmissions or []:
        row = dict(edge or {})
        sender = row.get("sender")
        receiver = row.get("receiver")
        transmission_strength = float(row.get("transmission_strength", row.get("edge_transmission_strength", 0.0)) or 0.0)
        source_pi = float(row.get("source_pi", 0.0) or 0.0)
        target_rho = float(row.get("target_rho", 0.0) or 0.0)
        source_h_final = float(row.get("source_h_final", row.get("source_aggregate_risk", source_pi)) or 0.0)
        target_h_final = float(row.get("target_h_final", row.get("target_aggregate_risk", target_rho)) or 0.0)
        source_rho = float(
            row.get(
                "source_rho",
                infer_rho_from_h_pi(source_h_final, source_pi),
            )
            or 0.0
        )
        target_pi = float(
            row.get(
                "target_pi",
                infer_pi_from_h_rho(target_h_final, target_rho),
            )
            or 0.0
        )
        source_aggregate_risk = float(
            row.get("source_aggregate_risk", source_h_final) or 0.0
        )
        target_aggregate_risk = float(
            row.get("target_aggregate_risk", target_h_final) or 0.0
        )
        normalized_row = {
            "sender": sender,
            "receiver": receiver,
            "transmission_strength": transmission_strength,
            "source_pi": source_pi,
            "source_rho": source_rho,
            "target_pi": target_pi,
            "target_rho": target_rho,
            "source_h_final": source_h_final,
            "target_h_final": target_h_final,
            "source_aggregate_risk": source_aggregate_risk,
            "target_aggregate_risk": target_aggregate_risk,
            "aggregate_risk": {
                "source": source_aggregate_risk,
                "target": target_aggregate_risk,
            },
            "source_infection_penalty": float(row.get("source_infection_penalty", source_rho) or 0.0),
        }
        for legacy_key in row.keys():
            if legacy_key not in normalized_row:
                normalized_row[legacy_key] = row[legacy_key]
        normalized.append(normalized_row)
    return normalized


def edge_set(graph_dict) -> set[tuple[int, int]]:
    rows = set()
    for edge in graph_dict.get("edges", []) or []:
        if len(edge) >= 2:
            rows.add((int(edge[0]), int(edge[1])))
    return rows


def edge_transmission(transmission_matrix, src: int, dst: int) -> float:
    # An unobserved realized edge contributes no measured transmission.
    return safe_edge_t(transmission_matrix, src, dst, default_if_missing=0.0)


def edge_delta(graph_dict, agents, transmission_matrix, beta: float, src: int, dst: int) -> float:
    if src not in agents:
        return 0.0
    pi_j = float(getattr(agents[src], "predicted_prob", getattr(agents[src], "pi", 0.0)) or 0.0)
    rho_i = float(getattr(agents.get(dst), "rho", 0.0) or 0.0) if dst in agents else 0.0
    edge_t = edge_transmission(transmission_matrix, src, dst)
    return (1.0 - pi_j) - float(beta) * edge_t * pi_j * (1.0 + rho_i)


def compute_edge_quality_metrics(graph_dict, agents, transmission_matrix, beta: float) -> dict[str, float]:
    edges = edge_set(graph_dict)
    num_nodes = len(graph_dict.get("nodes", [])) or len(agents)
    if not edges:
        isolated = num_nodes if num_nodes > 0 else 0
        return {
            "effective_edge_ratio": 0.0,
            "isolated_node_ratio": (isolated / num_nodes) if num_nodes else 0.0,
            "edge_density": 0.0,
            "avg_in_degree": 0.0,
            "avg_out_degree": 0.0,
        }

    effective_count = 0
    in_degree = {idx: 0 for idx in range(num_nodes)}
    out_degree = {idx: 0 for idx in range(num_nodes)}
    for src, dst in edges:
        if edge_delta(graph_dict, agents, transmission_matrix, beta, src, dst) > 0.0:
            effective_count += 1
        if src in out_degree:
            out_degree[src] += 1
        if dst in in_degree:
            in_degree[dst] += 1

    isolated = sum(1 for idx in range(num_nodes) if in_degree.get(idx, 0) + out_degree.get(idx, 0) == 0)
    max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    return {
        "effective_edge_ratio": effective_count / len(edges),
        "isolated_node_ratio": isolated / num_nodes if num_nodes else 0.0,
        "edge_density": len(edges) / max_edges,
        "avg_in_degree": sum(in_degree.values()) / num_nodes if num_nodes else 0.0,
        "avg_out_degree": sum(out_degree.values()) / num_nodes if num_nodes else 0.0,
    }


def compute_topology_change_rate(current_edges: set[tuple[int, int]], previous_edges: set[tuple[int, int]] | None) -> float:
    if previous_edges is None:
        return 0.0
    return len(current_edges.symmetric_difference(previous_edges)) / (len(previous_edges) + 1e-8)


def classify_edge_sets(graph_dict, agents, transmission_matrix, beta: float) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    edges = edge_set(graph_dict)
    risky_edges = {
        edge for edge in edges
        if edge_delta(graph_dict, agents, transmission_matrix, beta, edge[0], edge[1]) < 0.0
    }
    benign_edges = edges - risky_edges
    return edges, risky_edges, benign_edges


def compute_final_edge_mechanism_metrics(
    *,
    initial_edges: set[tuple[int, int]],
    initial_risky_edges: set[tuple[int, int]],
    initial_benign_edges: set[tuple[int, int]],
    final_edges: set[tuple[int, int]],
    agents,
    graph_dict,
    transmission_matrix,
    beta: float,
) -> dict[str, float]:
    new_edges = final_edges - initial_edges
    effective_new = {
        edge for edge in new_edges
        if edge_delta(graph_dict, agents, transmission_matrix, beta, edge[0], edge[1]) > 0.0
    }
    return {
        "risky_edge_removal_rate": len(initial_risky_edges - final_edges) / (len(initial_risky_edges) + 1e-8),
        "benign_edge_preservation_rate": len(initial_benign_edges & final_edges) / (len(initial_benign_edges) + 1e-8),
        "new_effective_edge_ratio": len(effective_new) / (len(new_edges) + 1e-8),
    }


def summarize_action_events(latest_events) -> dict[str, float]:
    prune_count = 0
    seek_success_count = 0
    seek_blocked_count = 0
    self_reflect_trigger_count = 0
    self_reflect_changed_count = 0
    self_reflect_accepted_count = 0
    self_reflect_effective_count = 0
    self_reflect_risk_reduced_count = 0
    self_reflect_response_changed_count = 0
    self_reflect_role_changed_count = 0
    self_reflect_fitness_improved_count = 0
    self_reflect_blocked_count = 0
    targeted_self_evaluation_count = 0
    targeted_self_retained_count = 0
    targeted_self_delayed_pi_reduction_sum = 0.0
    rep_gain_sum = 0.0
    seek_gain_sum = 0.0
    for agent_events in latest_events or []:
        for ev in agent_events.get("events", []):
            event_type = ev.get("type")
            edge_delta_raw = float(ev.get("edge_delta_raw", ev.get("delta", 0.0)) or 0.0)
            if event_type == "prune":
                prune_count += 1
                rep_gain_sum += max(0.0, -edge_delta_raw)
            elif event_type == "seek_connect":
                seek_success_count += 1
                seek_gain_sum += max(0.0, edge_delta_raw)
            elif event_type == "seek_blocked":
                seek_blocked_count += 1
            elif event_type == "self_reflect":
                self_reflect_trigger_count += 1
                if bool(ev.get("changed", False)):
                    self_reflect_changed_count += 1
                accepted = bool(ev.get("accepted", False))
                if accepted:
                    self_reflect_accepted_count += 1
                baseline_u = float(ev.get("baseline_structural_fitness", 0.0) or 0.0)
                candidate_u = float(ev.get("candidate_structural_fitness", 0.0) or 0.0)
                fitness_improved = candidate_u > baseline_u + 1e-8
                if accepted and (
                    bool(ev.get("risk_reduced", False))
                    or bool(ev.get("response_changed", False))
                    or fitness_improved
                ):
                    self_reflect_effective_count += 1
                if bool(ev.get("risk_reduced", False)):
                    self_reflect_risk_reduced_count += 1
                if bool(ev.get("response_changed", False)):
                    self_reflect_response_changed_count += 1
                if bool(ev.get("role_changed", False)):
                    self_reflect_role_changed_count += 1
                if fitness_improved:
                    self_reflect_fitness_improved_count += 1
            elif event_type == "self_reflect_blocked":
                self_reflect_blocked_count += 1
            elif event_type == "self_reflect_evaluation":
                targeted_self_evaluation_count += 1
                accepted = bool(ev.get("accepted", False))
                risk_reduced = bool(ev.get("risk_reduced", False))
                if accepted:
                    targeted_self_retained_count += 1
                    self_reflect_accepted_count += 1
                    targeted_self_delayed_pi_reduction_sum += max(
                        0.0,
                        float(ev.get("delayed_pi_reduction", 0.0) or 0.0),
                    )
                if accepted and risk_reduced:
                    self_reflect_effective_count += 1
                if risk_reduced:
                    self_reflect_risk_reduced_count += 1

    seek_attempt_count = seek_success_count + seek_blocked_count
    governance_attempt_count = prune_count + seek_attempt_count + self_reflect_trigger_count
    governance_effective_count = prune_count + seek_success_count + self_reflect_effective_count
    return {
        "prune_count": prune_count,
        "seek_attempt_count": seek_attempt_count,
        "seek_success_count": seek_success_count,
        "seek_blocked_count": seek_blocked_count,
        "rewiring_success_rate": seek_success_count / (seek_attempt_count + 1e-8),
        "rep_gain_sum": rep_gain_sum,
        "seek_gain_sum": seek_gain_sum,
        "governance_action_count": governance_attempt_count,
        "governance_attempt_count": governance_attempt_count,
        "governance_effective_count": governance_effective_count,
        "self_reflect_count": self_reflect_trigger_count,
        "self_reflect_trigger_count": self_reflect_trigger_count,
        "self_reflect_changed_count": self_reflect_changed_count,
        "self_reflect_accepted_count": self_reflect_accepted_count,
        "self_reflect_attempt_count": self_reflect_trigger_count,
        "self_reflect_effective_count": self_reflect_effective_count,
        "self_reflect_risk_reduced_count": self_reflect_risk_reduced_count,
        "self_reflect_response_changed_count": self_reflect_response_changed_count,
        "self_reflect_role_changed_count": self_reflect_role_changed_count,
        "self_reflect_fitness_improved_count": self_reflect_fitness_improved_count,
        "self_reflect_blocked_count": self_reflect_blocked_count,
        "targeted_self_evaluation_count": targeted_self_evaluation_count,
        "targeted_self_retained_count": targeted_self_retained_count,
        "targeted_self_delayed_pi_reduction_sum": targeted_self_delayed_pi_reduction_sum,
    }


def _self_reflection_generation_context(config: ExperimentConfig) -> str:
    mode = get_functional_ablation_spec(
        getattr(config.egt, "functional_ablation_mode", "full")
    ).self_reflection_mode
    return {
        "targeted_deferred_intrinsic": "next_regular_generation_no_extra_call",
        "independent_resample_intrinsic": "question_options_independent_resample_no_peer_v1",
        "response_only_counterfactual": "own_previous_response_counterfactual_no_peer_v1",
        "response_only_intrinsic": "own_previous_response_no_peer_v1",
        "paper_aligned_synchronous": "paper_current_context_and_risk_shared_snapshot_v1",
    }.get(
        mode,
        (
            "question_only_no_peer_v1"
            if bool(getattr(config.egt, "enable_peer_free_self_reflection", False))
            else "peer_augmented_legacy"
        ),
    )


def export_experiment_results(history_stats, env, output_path: str = "experiment_results.json", extra_results=None):
    """Export history, final summary, and token usage to JSON."""
    final_stats = history_stats[-1] if history_stats else {}
    final_eval = dict(getattr(env, "latest_final_eval", {}) or {})
    final_answer_eval = dict(getattr(env, "latest_final_answer_eval", {}) or {})
    answer_choice_analysis = dict(getattr(env, "latest_answer_choice_analysis", {}) or {})
    consensus_analysis = dict(getattr(env, "latest_consensus_analysis", {}) or {})
    token_usage = dict(getattr(env, "token_usage", {}))
    agents = list(getattr(env, "agents", {}).values())
    total_prompt_tokens = (
        token_usage.get("generation_prompt", 0)
        + token_usage.get("rep_judge_prompt", 0)
        + token_usage.get("judge_prompt", 0)
    )
    total_completion_tokens = (
        token_usage.get("generation_completion", 0)
        + token_usage.get("rep_judge_completion", 0)
        + token_usage.get("judge_completion", 0)
    )
    final_edges = edge_set(getattr(env, "graph_dict", {}))
    final_transmission_matrix = getattr(env, "latest_transmission_matrix", [])
    final_edge_quality = compute_edge_quality_metrics(
        getattr(env, "graph_dict", {}),
        getattr(env, "agents", {}),
        final_transmission_matrix,
        _current_edge_beta(env),
    )
    initial_edges = getattr(env, "metric_initial_edges", set())
    initial_risky_edges = getattr(env, "metric_initial_risky_edges", set())
    initial_benign_edges = getattr(env, "metric_initial_benign_edges", set())
    final_edge_mechanism = compute_final_edge_mechanism_metrics(
        initial_edges=initial_edges,
        initial_risky_edges=initial_risky_edges,
        initial_benign_edges=initial_benign_edges,
        final_edges=final_edges,
        agents=getattr(env, "agents", {}),
        graph_dict=getattr(env, "graph_dict", {}),
        transmission_matrix=final_transmission_matrix,
        beta=_current_edge_beta(env),
    )
    first_rho = float(history_stats[0].get("avg_rho", history_stats[0].get("avg_h_final", 0.0)) or 0.0) if history_stats else 0.0
    last_rho = float(final_stats.get("avg_rho", final_stats.get("avg_h_final", 0.0)) or 0.0)
    rho_reduction = first_rho - last_rho
    first_h = float(history_stats[0].get("avg_h_final", 0.0) or 0.0) if history_stats else 0.0
    last_h = float(final_stats.get("avg_h_final", 0.0) or 0.0)
    aggregate_risk_reduction = first_h - last_h
    total_governance_attempts = int(getattr(env, "metric_total_governance_attempts", 0))
    total_governance_effective_actions = int(getattr(env, "metric_total_governance_effective_actions", 0))
    if rho_reduction > 1e-8 and total_governance_effective_actions > 0:
        action_efficiency = rho_reduction / total_governance_effective_actions
    else:
        action_efficiency = None
    final_stats.update(
        {
            "post_action_edge_count": len(final_edges),
            "collapse_flag": 1.0 if len(final_edges) == 0 else 0.0,
            "rho_area": sum(float(row.get("avg_rho", row.get("avg_h_final", 0.0)) or 0.0) for row in history_stats),
            "aggregate_risk_area": sum(float(row.get("avg_h_final", 0.0) or 0.0) for row in history_stats),
            "utility_area": sum(float(row.get("avg_fitness", 0.0) or 0.0) for row in history_stats),
            "edge_risk_area": sum(float(row.get("avg_edge_risk", 0.0) or 0.0) for row in history_stats),
            "adjusted_edge_risk_area": sum(float(row.get("avg_adjusted_edge_risk", 0.0) or 0.0) for row in history_stats),
            "rho_auc_legacy": sum(float(row.get("avg_rho", row.get("avg_h_final", 0.0)) or 0.0) for row in history_stats),
            "risk_auc_legacy": sum(float(row.get("avg_h_final", 0.0) or 0.0) for row in history_stats),
            "utility_auc_legacy": sum(float(row.get("avg_fitness", 0.0) or 0.0) for row in history_stats),
            "edge_risk_auc_legacy": sum(float(row.get("avg_edge_risk", 0.0) or 0.0) for row in history_stats),
            "rho_reduction": rho_reduction,
            "risk_reduction": aggregate_risk_reduction,
            "action_efficiency": action_efficiency,
            "per_round_rho": [float(row.get("avg_rho", row.get("avg_h_final", 0.0)) or 0.0) for row in history_stats],
            "total_prune_count": int(getattr(env, "metric_total_prune_count", 0)),
            "total_seek_attempts": int(getattr(env, "metric_total_seek_attempts", 0)),
            "total_seek_successes": int(getattr(env, "metric_total_seek_successes", 0)),
            "total_self_reflect_count": int(getattr(env, "metric_total_self_reflect_count", 0)),
            "total_governance_attempts": total_governance_attempts,
            "total_governance_effective_actions": total_governance_effective_actions,
            "total_generation_failure_count": sum(
                int(row.get("generation_failure_count", 0) or 0)
                for row in history_stats
            ),
            "overall_rewiring_success_rate": (
                int(getattr(env, "metric_total_seek_successes", 0))
                / (int(getattr(env, "metric_total_seek_attempts", 0)) + 1e-8)
            ),
            "edge_beta_used": float(getattr(env, "latest_edge_beta_used", 0.0) or 0.0),
            "edge_beta_raw": float(getattr(env, "latest_edge_beta_raw", 0.0) or 0.0),
            "edge_beta_capped": bool(getattr(env, "latest_edge_beta_capped", False)),
            "edge_risk_ratio_q85": float(getattr(env, "latest_edge_risk_ratio_q85", 0.0) or 0.0),
            "avg_edge_risk_ratio": float(getattr(env, "latest_avg_edge_risk_ratio", 0.0) or 0.0),
            "max_edge_risk_ratio": float(getattr(env, "latest_max_edge_risk_ratio", 0.0) or 0.0),
            "gold_answer": final_answer_eval.get("gold_answer"),
            "question_final_answer": final_answer_eval.get("question_final_answer"),
            "question_final_parsed_answer": final_answer_eval.get("question_final_parsed_answer"),
            "majority_vote_answer": final_answer_eval.get("majority_vote_answer"),
            "majority_vote_parsed_answer": final_answer_eval.get("majority_vote_parsed_answer"),
            "selector_answer": final_answer_eval.get("selector_answer"),
            "selector_accuracy": final_answer_eval.get("selector_accuracy"),
            "selector_status": final_answer_eval.get("selector_status"),
            "lowest_risk_agent_answer": final_answer_eval.get("lowest_risk_agent_answer"),
            "lowest_risk_agent_parsed_answer": final_answer_eval.get("lowest_risk_agent_parsed_answer"),
            "final_selection_method": final_answer_eval.get("final_selection_method"),
            "final_selection_uses_risk": final_answer_eval.get("final_selection_uses_risk"),
            "valid_final_answer_count": final_answer_eval.get("valid_final_answer_count"),
            "final_format_retry_triggered": final_answer_eval.get("final_format_retry_triggered"),
            "final_format_retry_succeeded": final_answer_eval.get("final_format_retry_succeeded"),
            "question_final_accuracy": final_answer_eval.get("question_final_accuracy"),
            "majority_vote_accuracy": final_answer_eval.get("majority_vote_accuracy"),
            "lowest_risk_agent_accuracy": final_answer_eval.get("lowest_risk_agent_accuracy"),
            "agent_answer_accuracy": final_answer_eval.get("agent_answer_accuracy"),
            "agent_answer_accuracy_valid_only": final_answer_eval.get("agent_answer_accuracy_valid_only"),
            "parse_fail_rate": final_answer_eval.get("parse_fail_rate"),
            "final_answer_accuracy": final_answer_eval.get("question_final_accuracy"),
            "final_agent_answer_accuracy": final_answer_eval.get("agent_answer_accuracy"),
            "final_agent_answer_accuracy_valid_only": final_answer_eval.get("agent_answer_accuracy_valid_only"),
            "final_answer_parse_fail_rate": final_answer_eval.get("parse_fail_rate"),
            "answer_choice_utility": float(answer_choice_analysis.get("avg_answer_choice_utility", 0.0) or 0.0),
            "consensus_answer": consensus_analysis.get("consensus_answer"),
            "consensus_ratio": float(consensus_analysis.get("consensus_ratio", 0.0) or 0.0),
            "consensus_risk": float(consensus_analysis.get("consensus_risk", 0.0) or 0.0),
            "consensus_breaker_triggered": bool(consensus_analysis.get("consensus_breaker_triggered", False)),
            **final_edge_quality,
            **final_edge_mechanism,
        }
    )
    triggered_rows = [row for row in history_stats if bool(row.get("consensus_breaker_triggered", 0.0))]
    triggered_consensus_answer = triggered_rows[0].get("consensus_answer") if triggered_rows else None
    max_triggered_consensus_risk = max((float(row.get("consensus_risk", 0.0) or 0.0) for row in triggered_rows), default=0.0)
    final_stats["consensus_breaker_success"] = bool(
        triggered_rows
        and (
            float(consensus_analysis.get("consensus_risk", 0.0) or 0.0) + 1e-8 < max_triggered_consensus_risk
            or consensus_analysis.get("consensus_answer") != triggered_consensus_answer
        )
    )
    latest_risk_details = dict(getattr(env, "latest_prediction_details", {}).get("details", {}) or {})
    final_stats.update(_build_cumulative_risk_judge_audit(env))
    final_stats["online_risk_judge_tokens"] = int(token_usage.get("online_risk_judge_prompt", 0) or 0) + int(
        token_usage.get("online_risk_judge_completion", 0) or 0
    )
    final_stats["final_offline_judge_tokens"] = int(token_usage.get("final_judge_prompt", 0) or 0) + int(
        token_usage.get("final_judge_completion", 0) or 0
    )
    def _json_safe(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, set):
            return [_json_safe(v) for v in sorted(value)]
        return value

    def _sanitize_secrets(value):
        if isinstance(value, dict):
            sanitized = {}
            for k, v in value.items():
                key_lower = str(k).lower()
                if "key" in key_lower or key_lower == "url" or key_lower.endswith("_url"):
                    sanitized[str(k)] = "***REDACTED***" if v else v
                else:
                    sanitized[str(k)] = _sanitize_secrets(v)
            return sanitized
        if isinstance(value, list):
            return [_sanitize_secrets(v) for v in value]
        if isinstance(value, tuple):
            return [_sanitize_secrets(v) for v in value]
        return value

    config_snapshot = _json_safe(_sanitize_secrets(asdict(getattr(env, "config", build_default_config()))))

    agent_history = getattr(env, "agent_history", [])
    tag_frequency_history = getattr(env, "tag_frequency_history", [])
    strategy_stats_history = getattr(env, "strategy_stats_history", [])
    action_history = getattr(env, "action_history", [])
    response_update_history = getattr(env, "response_update_history", [])
    graph_history = getattr(env, "graph_history", [])
    transmission_history = [
        {
            **entry,
            "edges": normalize_edge_transmission_records(entry.get("edges", [])),
        }
        for entry in (getattr(env, "transmission_history", []) or [])
    ]
    graph_summary = {
        "num_agents": len(agents),
        "final_edge_count": len(getattr(env, "graph_dict", {}).get("edges", [])),
        "question_text": getattr(env, "graph_dict", {}).get("question_text"),
    }

    export_payload = {
        "meta": {
            "risk_estimator": "llm_judge_atomic_observer",
            "risk_estimator_mode": str(getattr(env.config.egt, "risk_estimator_mode", "judge_observer")),
            "risk_judge_used": True,
            "final_scoring_basis": "llm_judge_atomic_pi_rho_T_risk_fitness",
            "invalid_response_fallback": bool(
                getattr(env.config.egt, "enable_invalid_response_fallback", True)
            ),
            "response_update_policy": (
                "invalid_only_preserve_previous_valid_v1"
                if bool(getattr(env.config.egt, "enable_invalid_response_fallback", True))
                else "unconditional_candidate_commit_legacy"
            ),
            "online_risk_prompt_protocol_version": (
                "reference_free_argument_compact_split_context_v4"
                if bool(getattr(env.config.egt, "deduplicate_online_risk_prompt_context", True))
                else "duplicated_context_v1"
            ),
            "risk_judge_prompt_version": env.RISK_JUDGE_PROMPT_VERSION,
            "transmission_judge_prompt_version": env.TRANSMISSION_JUDGE_PROMPT_VERSION,
            "risk_context_has_peer_messages": False,
            "online_judge_has_gold": False,
            "transmission_uses_before_after": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).semantic_transmission_enabled,
            "transmission_sender_risk_visible": False,
            "transmission_judge_used": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).semantic_transmission_enabled,
            "functional_ablation_mode": str(
                getattr(config.egt, "functional_ablation_mode", "full")
            ),
            "functional_ablation_stage": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).cumulative_stage,
            "self_reflection_mode": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).self_reflection_mode,
            "strategy_evolution_enabled": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).strategy_evolution_enabled,
            "topology_governance_enabled": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).topology_governance_enabled,
            "reuse_unchanged_claim_observations": bool(
                getattr(env.config.egt, "reuse_unchanged_claim_observations", True)
            ),
            "online_risk_observer_oracle_free": bool(
                getattr(env.config.egt, "online_risk_observer_oracle_free", True)
            ),
            "online_risk_observer_uses_gold": False,
            "offline_final_evaluator_uses_gold": True,
            "offline_final_component_judge": True,
            "final_component_judge_prompt_version": env.FINAL_COMPONENT_JUDGE_VERSION,
            "final_component_scoring_basis": env.FINAL_COMPONENT_SCORING_BASIS,
            "method_protocol_version": env.METHOD_PROTOCOL_VERSION,
            "peer_message_prompt_version": env.PEER_MESSAGE_PROMPT_VERSION,
            "transmission_timing": "sender_t_minus_1_to_receiver_t",
            "unknown_transmission_policy": (
                "realized_invalid_or_unobserved_T0;prospective_seek_screening_T1"
            ),
            "transmission_claim_risk_threshold": float(
                env.config.egt.transmission_claim_risk_threshold
            ),
            "initial_state_risk_observer_passes": 1,
            "initial_state_message_policy": "raw_output_only",
            "initial_state_transmission_identifiable": False,
            "initial_state_rho_mode": "not_inferred_before_peer_message_consumption",
            "pre_governance_risk_observer_passes_per_round": 1,
            "post_governance_risk_observer_policy": "rerun_only_if_agent_response_changed",
            "max_main_risk_observer_passes_per_round": max(
                (
                    int(row.get("online_risk_observer_passes_this_round", 0) or 0)
                    for row in history_stats
                ),
                default=0,
            ),
            "risk_observer_pass_count_semantics": (
                "actual_full_state_estimator_invocations_including_self_candidates"
            ),
            "risk_observer_invocation_count": int(
                getattr(env, "risk_observer_invocation_count", 0) or 0
            ),
            "response_refresh_interval": int(env.config.egt.response_refresh_interval),
            "edge_beta_definition": (
                "configured_beta_if_no_valid_ratios;"
                "1_if_q85_le_eps_else_min(q85_edge_risk_ratio,cap)"
            ),
            "selection_alpha_schedule": "linear",
            "strategy_mutation_operator": "single_bit_flip",
            "fermi_candidate_pool": "all_other_agents",
            "online_consensus_uses_injection_wrong_label": False,
            "actual_response_generation_rounds": int(getattr(env, "response_generation_round_count", 0)),
            "round_indexing": "gen_0_based_round_1_based",
            "risk_area_scope": "generated_rounds_only_excludes_injected_round_0",
            "final_answer_selection": "majority_vote_and_gold_blind_deepseek_selector_v1",
            "final_answer_selection_uses_risk": False,
            "final_format_retry_enabled": False,
            "self_reflection_generation_context": _self_reflection_generation_context(
                env.config
            ),
        },
        "num_generations": len(history_stats),
        "config_snapshot": config_snapshot,
        "graph_summary": graph_summary,
        "history_stats": history_stats,
        "agent_history": agent_history,
        "tag_frequency_history": tag_frequency_history,
        "strategy_stats_history": strategy_stats_history,
        "action_history": action_history,
        "response_update_history": response_update_history,
        "graph_history": graph_history,
        "transmission_history": transmission_history,
        "initial_risk_state": (
            env.initial_risk_state.to_legacy_details(size=len(env.agents))
            if env.initial_risk_state is not None
            else None
        ),
        "final_summary": final_stats,
        "final_eval": final_eval,
        "final_answer_eval": final_answer_eval,
        "answer_choice_analysis": answer_choice_analysis,
        "consensus_analysis": consensus_analysis,
        "online_prompt_samples": list(getattr(env, "latest_online_prompt_samples", []) or []),
        "token_usage": token_usage,
        "rep_judge_cache_summary": {
            "cache_size": len(getattr(env, "rep_judge_cache", {})),
            "cache_hits": int(getattr(env, "rep_judge_cache_hits", 0)),
        },
        "api_request_summary": env.get_api_request_stats(),
        "risk_judge_audit": _build_cumulative_risk_judge_audit(env),
        "token_usage_summary": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "gold_answer": final_answer_eval.get("gold_answer"),
        "question_final_answer": final_answer_eval.get("question_final_answer"),
        "question_final_parsed_answer": final_answer_eval.get("question_final_parsed_answer"),
        "majority_vote_answer": final_answer_eval.get("majority_vote_answer"),
        "majority_vote_parsed_answer": final_answer_eval.get("majority_vote_parsed_answer"),
        "selector_answer": final_answer_eval.get("selector_answer"),
        "selector_parsed_answer": final_answer_eval.get("selector_parsed_answer"),
        "selector_accuracy": final_answer_eval.get("selector_accuracy"),
        "selector_status": final_answer_eval.get("selector_status"),
        "selector_model": final_answer_eval.get("selector_model"),
        "selector_error": final_answer_eval.get("selector_error"),
        "lowest_risk_agent_answer": final_answer_eval.get("lowest_risk_agent_answer"),
        "lowest_risk_agent_parsed_answer": final_answer_eval.get("lowest_risk_agent_parsed_answer"),
        "final_selection_method": final_answer_eval.get("final_selection_method"),
        "final_selection_uses_risk": final_answer_eval.get("final_selection_uses_risk"),
        "valid_final_answer_count": final_answer_eval.get("valid_final_answer_count"),
        "final_format_retry_triggered": final_answer_eval.get("final_format_retry_triggered"),
        "final_format_retry_succeeded": final_answer_eval.get("final_format_retry_succeeded"),
        "per_agent_final_answers": final_answer_eval.get("per_agent_final_answers", []),
        "question_final_accuracy": final_answer_eval.get("question_final_accuracy"),
        "majority_vote_accuracy": final_answer_eval.get("majority_vote_accuracy"),
        "lowest_risk_agent_accuracy": final_answer_eval.get("lowest_risk_agent_accuracy"),
        "agent_answer_accuracy": final_answer_eval.get("agent_answer_accuracy"),
        "agent_answer_accuracy_valid_only": final_answer_eval.get("agent_answer_accuracy_valid_only"),
        "parse_fail_rate": final_answer_eval.get("parse_fail_rate"),
        "answer_choice_utility": float(answer_choice_analysis.get("avg_answer_choice_utility", 0.0) or 0.0),
        "consensus_answer": consensus_analysis.get("consensus_answer"),
        "consensus_ratio": float(consensus_analysis.get("consensus_ratio", 0.0) or 0.0),
        "consensus_risk": float(consensus_analysis.get("consensus_risk", 0.0) or 0.0),
        "consensus_breaker_triggered": bool(consensus_analysis.get("consensus_breaker_triggered", False)),
        "consensus_breaker_success": bool(final_stats.get("consensus_breaker_success", False)),
    }

    if extra_results:
        export_payload.update(extra_results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(_sanitize_secrets(export_payload)), handle, ensure_ascii=False, indent=2)

    print(f"[Export] Saved experiment results to {output_path}")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def _quantile(values, q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _trimmed_mean(values, trim_ratio: float = 0.1) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    n = len(xs)
    k = int(n * max(0.0, min(0.49, float(trim_ratio))))
    if 2 * k >= n:
        return sum(xs) / max(1, n)
    core = xs[k : n - k]
    return sum(core) / max(1, len(core))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _current_edge_beta(env: EvolutionEnvironment, config: ExperimentConfig | None = None) -> float:
    fallback = getattr((config or env.config).egt, "fitness_penalty_beta", 1.0)
    value = getattr(env, "latest_edge_beta_used", fallback)
    return float(fallback if value is None else value)


def _annealed_noise(noise_start: float, gen_idx: int, total_gens: int) -> float:
    if total_gens <= 1:
        return max(0.0, float(noise_start))
    progress = max(0.0, min(1.0, gen_idx / (total_gens - 1)))
    return max(0.0, float(noise_start) * (1.0 - progress))


def apply_fixed_rule_parameter_calibration(config: ExperimentConfig, env: EvolutionEnvironment) -> dict:
    """
    Fixed-rule, no-new-knob calibration:
    1) keep Fermi selection_alpha_start/selection_alpha_end on the configured selection schedule
    2) verify_cost from mean risk-reduction benefit
    3) search_cost from existing ratio search_cost/verify_cost
    4) mutation_start/mutation_end and persona_mutation_rate from avg tag count
    5) exploration_noise from 0.1 * mean(|edge_delta|)
    """
    edge_records = list(getattr(env, "latest_edge_transmissions", []) or [])
    if not edge_records:
        return {"applied": False, "reason": "no edge transmission records"}

    beta = float(config.egt.fitness_penalty_beta)
    edge_delta_abs_values = []
    verify_benefit_values = []

    for edge in edge_records:
        source_risk = max(0.0, min(1.0, float(edge.get("source_pi", edge.get("source_h_final", 0.0)) or 0.0)))
        target_risk = max(0.0, min(1.0, float(edge.get("target_rho", 0.0) or 0.0)))
        transmission = float(edge.get("transmission_strength", 0.0) or 0.0)
        if transmission <= 1e-8:
            continue
        if source_risk <= 1e-6:
            continue

        denom = transmission * source_risk * (1.0 + target_risk)
        if denom <= 1e-8:
            continue

        edge_delta = (1.0 - source_risk) - beta * transmission * source_risk * (1.0 + target_risk)
        edge_delta_abs_values.append(abs(edge_delta))
        verify_benefit_values.append(max(0.0, -edge_delta))

    if not edge_delta_abs_values:
        return {"applied": False, "reason": "insufficient valid edge samples"}
    selection_alpha_start = float(config.egt.selection_alpha_start)
    selection_alpha_end = float(config.egt.selection_alpha_end)
    default_verify_cost = float(config.egt.verify_cost)
    benefit_eps = 1e-8

    # Robust estimate: trimmed mean (drop top/bottom 10%) to reduce extreme-event distortion.
    robust_verify_benefit = _trimmed_mean(verify_benefit_values, trim_ratio=0.1)
    avg_abs_edge_delta = sum(edge_delta_abs_values) / max(1, len(edge_delta_abs_values))
    exploration_noise = max(0.0, min(0.5, 0.1 * avg_abs_edge_delta))

    avg_tag_count = (
        sum(len(agent.tags) for agent in env.agents.values()) / max(1, len(env.agents))
        if env.agents else 1.0
    )
    avg_tag_count = max(1.0, float(avg_tag_count))
    mutation_start = max(0.001, min(1.0, 1.0 / avg_tag_count))
    persona_mutation_rate = max(0.01, min(1.0, 1.0 / avg_tag_count))

    if robust_verify_benefit <= benefit_eps:
        return {
            "applied": False,
            "calibration_reason": "zero_benefit_fallback",
            "edge_samples": len(edge_delta_abs_values),
            "selection_alpha_start": selection_alpha_start,
            "selection_alpha_end": selection_alpha_end,
            "verify_cost": float(config.egt.verify_cost),
            "verify_benefit_trimmed_mean": robust_verify_benefit,
            "search_cost": float(config.egt.search_cost),
            "search_verify_ratio": MEASURED_SEARCH_VERIFY_COST_RATIO,
            "search_verify_ratio_source": SEARCH_VERIFY_RATIO_SOURCE,
            "exploration_noise": float(config.egt.exploration_noise),
            "mutation_start": float(config.egt.mutation_start),
            "mutation_end": float(config.egt.mutation_end),
            "mutation_end_start_ratio": MUTATION_END_START_RATIO,
            "persona_mutation_rate": float(config.egt.persona_mutation_rate),
        }

    config.egt.selection_alpha_start = selection_alpha_start
    config.egt.selection_alpha_end = selection_alpha_end
    # Anchor to the existing verify_cost scale to avoid sudden magnitude jumps.
    config.egt.verify_cost = math.sqrt(
        max(1e-12, default_verify_cost) * max(1e-12, robust_verify_benefit)
    )
    config.egt.exploration_noise = exploration_noise
    config.egt.mutation_start = mutation_start
    config.egt.persona_mutation_rate = persona_mutation_rate
    config.sync_derived_costs()
    search_cost = config.egt.search_cost
    mutation_end = config.egt.mutation_end
    verify_cost = float(config.egt.verify_cost)

    return {
        "applied": True,
        "calibration_reason": None,
        "edge_samples": len(edge_delta_abs_values),
        "selection_alpha_start": selection_alpha_start,
        "selection_alpha_end": selection_alpha_end,
        "verify_cost": verify_cost,
        "verify_benefit_trimmed_mean": robust_verify_benefit,
        "search_cost": search_cost,
        "search_verify_ratio": MEASURED_SEARCH_VERIFY_COST_RATIO,
        "search_verify_ratio_source": SEARCH_VERIFY_RATIO_SOURCE,
        "exploration_noise": exploration_noise,
        "mutation_start": mutation_start,
        "mutation_end": mutation_end,
        "mutation_end_start_ratio": MUTATION_END_START_RATIO,
        "persona_mutation_rate": persona_mutation_rate,
    }


def _build_model_token_stats(token_usage: dict[str, int], config) -> dict[str, Any]:
    model_names = {
        "generation": str(getattr(config.llm, "generation_model", "generation")),
        "rep_judge": str(getattr(config.llm, "judge_model", "judge")),
        "judge": str(getattr(config.llm, "judge_model", "judge")),
    }
    stats: dict[str, dict[str, Any]] = {}
    for prefix in ("generation", "rep_judge", "judge"):
        model_name = model_names[prefix]
        entry = stats.setdefault(
            model_name,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        prompt_tokens = int(token_usage.get(f"{prefix}_prompt", 0) or 0)
        completion_tokens = int(token_usage.get(f"{prefix}_completion", 0) or 0)
        if prefix == "judge":
            prompt_tokens -= int(token_usage.get("final_selector_prompt", 0) or 0)
            completion_tokens -= int(token_usage.get("final_selector_completion", 0) or 0)
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["total_tokens"] += prompt_tokens + completion_tokens
    selector_prompt = int(token_usage.get("final_selector_prompt", 0) or 0)
    selector_completion = int(token_usage.get("final_selector_completion", 0) or 0)
    if selector_prompt or selector_completion:
        entry = stats.setdefault("deepseek-v4-pro", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        entry["prompt_tokens"] += selector_prompt
        entry["completion_tokens"] += selector_completion
        entry["total_tokens"] += selector_prompt + selector_completion
    return stats


def _build_cumulative_risk_judge_audit(env) -> dict[str, Any]:
    totals = dict(getattr(env, "risk_judge_audit_totals", {}) or {})
    call_count = max(1, int(totals.get("risk_judge_call_count", 0) or 0))
    return {
        **totals,
        "risk_claim_judge_tokens": int(totals.get("risk_claim_judge_prompt_tokens", 0) or 0)
        + int(totals.get("risk_claim_judge_completion_tokens", 0) or 0),
        "transmission_judge_tokens": int(totals.get("transmission_judge_prompt_tokens", 0) or 0)
        + int(totals.get("transmission_judge_completion_tokens", 0) or 0),
        "risk_judge_capped_rate": float(totals.get("risk_judge_capped_count", 0) or 0) / call_count,
        "risk_judge_invalid_rate": float(totals.get("risk_judge_invalid_count", 0) or 0) / call_count,
    }


def _extract_gold_answer(graph_dict: dict[str, Any], valid_letters: list[str] | None = None) -> str | None:
    for key in (
        "gold_answer",
        "ground_truth",
        "ground_truth_answer",
        "correct_answer",
        "answer",
        "label",
        "target",
    ):
        parsed = parse_mmlupro_choice_answer(graph_dict.get(key), model_name=None, valid_letters=valid_letters)
        if parsed:
            return parsed
    return None


def _select_final_answer(
    env: EvolutionEnvironment,
    config: ExperimentConfig,
    agent_responses: dict[str, str],
    valid_letters: list[str],
) -> dict[str, Any]:
    """Select from final responses without sending the ground truth to the model."""
    parsed_answers = [
        parse_mmlupro_choice_answer(response, valid_letters=valid_letters)
        for response in agent_responses.values()
    ]
    if parsed_answers and all(answer is not None and answer == parsed_answers[0] for answer in parsed_answers):
        return {"answer": parsed_answers[0], "status": "unanimous_consensus", "model": None, "error": None}

    model_name = "deepseek-v4-pro"
    question = str(env.graph_dict.get("question") or env.graph_dict.get("question_text") or "").strip()
    choices = env.graph_dict.get("choices") or []
    choice_lines = [f"{letter}. {choice}" for letter, choice in zip(valid_letters, choices)]
    response_lines = [f"Agent {agent_id}: {response}" for agent_id, response in agent_responses.items()]
    messages = [
        {
            "role": "system",
            "content": (
                "Select the best answer to the multiple-choice question from the agents' final responses. "
                "Return only JSON with one field: answer, containing one option letter. "
                "Do not assume a majority is correct."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n" + question + "\n\nOptions:\n" + "\n".join(choice_lines)
                + "\n\nFinal agent responses:\n" + "\n\n".join(response_lines)
            ),
        },
    ]
    try:
        route = resolve_model(model_name, role="judge")
        if not route["url"]:
            raise ValueError("Missing final selector HTTPS URL; set JUDGE_BASE_URL.")
        api_key = (
            config.llm.judge_key
            if config.llm.judge_model == model_name and config.llm.judge_key
            else route["key"]
        )
        response = env._call_llm_with_retry(
            "final_answer_selector",
            lambda: env._deepseek_chat_completion(
                base_url=str(route["url"]),
                api_key=str(api_key or ""),
                model_name=model_name,
                messages=messages,
                max_tokens=128,
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout_seconds=120,
            ),
        )
        usage = env._extract_usage(response)
        env._accumulate_usage(usage, "judge_prompt", "judge_completion")
        env._accumulate_usage(usage, "final_selector_prompt", "final_selector_completion")
        payload = json.loads(env._extract_chat_content(response))
        if not isinstance(payload, dict):
            raise ValueError("selector response must be a JSON object")
        answer = parse_mmlupro_choice_answer(payload.get("answer"), valid_letters=valid_letters)
        if answer is None:
            raise ValueError("selector returned no valid option letter")
        return {"answer": answer, "status": "selected", "model": model_name, "error": None}
    except Exception as exc:
        return {"answer": None, "status": "failed", "model": model_name, "error": f"{type(exc).__name__}: {exc}"}


def _build_final_answer_eval(env, config) -> dict[str, Any]:
    nodes = env.graph_dict.get("nodes", [])
    agent_responses: dict[str, str] = {}
    for node_id in sorted(env.agents.keys()):
        response_text = ""
        if isinstance(nodes, list) and node_id < len(nodes) and isinstance(nodes[node_id], dict):
            response_text = str(nodes[node_id].get("response") or "").strip()
        agent_responses[str(int(node_id))] = response_text

    valid_letters = choice_letters_from_choices(env.graph_dict.get("choices"))
    gold_answer = _extract_gold_answer(env.graph_dict, valid_letters)
    answer_eval = evaluate_final_agent_answers(
        agent_responses,
        gold_answer,
        model_name=str(getattr(config.llm, "generation_model", "") or ""),
        valid_letters=valid_letters,
        parser=parse_mmlupro_choice_answer,
    )
    per_agent_records = list(answer_eval.get("per_agent", []) or [])
    selector = _select_final_answer(env, config, agent_responses, valid_letters)
    per_agent_by_id = {int(item.get("agent_id", -1)): item for item in per_agent_records}

    lowest_risk_agent_id = None
    if env.agents:
        lowest_risk_agent_id = min(
            env.agents.keys(),
            key=lambda agent_id: (
                float(getattr(env.agents[agent_id], "h_final", 1.0) or 1.0),
                float(getattr(env.agents[agent_id], "rho", 1.0) or 1.0),
                int(agent_id),
            ),
        )

    majority_vote_parsed_answer = answer_eval.get("majority_answer")
    majority_vote_answer = None
    if majority_vote_parsed_answer is not None:
        for item in per_agent_records:
            if item.get("parsed_answer") == majority_vote_parsed_answer:
                majority_vote_answer = item.get("response")
                break

    lowest_risk_agent_answer = None
    lowest_risk_agent_parsed_answer = None
    lowest_risk_agent_accuracy = None
    if lowest_risk_agent_id is not None:
        lowest_item = per_agent_by_id.get(int(lowest_risk_agent_id), {})
        lowest_risk_agent_answer = lowest_item.get("response")
        lowest_risk_agent_parsed_answer = lowest_item.get("parsed_answer")
        if gold_answer and lowest_risk_agent_parsed_answer:
            lowest_risk_agent_accuracy = 1.0 if lowest_risk_agent_parsed_answer == gold_answer else 0.0

    question_final_answer = majority_vote_answer
    question_final_parsed_answer = majority_vote_parsed_answer
    if majority_vote_parsed_answer is not None:
        final_selection_method = "valid_final_majority_vote"
    else:
        final_selection_method = "no_valid_final_answer"
    question_final_accuracy = None
    majority_vote_accuracy = None
    if gold_answer and question_final_parsed_answer:
        question_final_accuracy = 1.0 if question_final_parsed_answer == gold_answer else 0.0
    if gold_answer and majority_vote_parsed_answer:
        majority_vote_accuracy = 1.0 if majority_vote_parsed_answer == gold_answer else 0.0
    selector_accuracy = None
    if gold_answer and selector["answer"]:
        selector_accuracy = 1.0 if selector["answer"] == gold_answer else 0.0

    per_agent_final_answers = [
        {
            "agent_id": int(item.get("agent_id", -1)),
            "response": item.get("response"),
            "parsed_answer": item.get("parsed_answer"),
            "correct": item.get("correct"),
        }
        for item in per_agent_records
    ]
    return {
        "gold_answer": gold_answer,
        "question_final_answer": question_final_answer,
        "question_final_parsed_answer": question_final_parsed_answer,
        "majority_vote_answer": majority_vote_answer,
        "majority_vote_parsed_answer": majority_vote_parsed_answer,
        "selector_answer": selector["answer"],
        "selector_parsed_answer": selector["answer"],
        "selector_accuracy": selector_accuracy,
        "selector_status": selector["status"],
        "selector_model": selector["model"],
        "selector_error": selector["error"],
        "lowest_risk_agent_id": int(lowest_risk_agent_id) if lowest_risk_agent_id is not None else None,
        "lowest_risk_agent_answer": lowest_risk_agent_answer,
        "lowest_risk_agent_parsed_answer": lowest_risk_agent_parsed_answer,
        "final_selection_method": final_selection_method,
        "final_selection_uses_risk": False,
        "valid_final_answer_count": len(answer_eval.get("agent_answers", []) or []),
        "final_format_retry_enabled": False,
        "final_format_retry_triggered": False,
        "final_format_retry_reason": "disabled",
        "final_format_retry_agent_id": None,
        "final_format_retry_response": None,
        "final_format_retry_parsed_answer": None,
        "final_format_retry_succeeded": False,
        "final_format_retry_error": None,
        "question_final_accuracy": question_final_accuracy,
        "majority_vote_accuracy": majority_vote_accuracy,
        "lowest_risk_agent_accuracy": lowest_risk_agent_accuracy,
        "agent_answer_accuracy": answer_eval.get("agent_answer_accuracy"),
        "agent_answer_accuracy_valid_only": answer_eval.get("agent_answer_accuracy_valid_only"),
        "parse_fail_rate": answer_eval.get("parse_fail_rate"),
        "per_agent_final_answers": per_agent_final_answers,
    }


def _evaluate_final_layer_with_judge(env, config) -> dict[str, Any]:
    question_text = (
        env.graph_dict.get("question")
        or env.graph_dict.get("question_text")
        or env.DEFAULT_QUESTION
    )
    gold_reference = env._build_offline_gold_reference()
    nodes = env.graph_dict.get("nodes", [])
    per_agent = []
    exact_response_cache: dict[str, dict[str, Any]] = {}
    exact_cache_hit_count = 0
    for node_id in sorted(env.agents.keys()):
        response_text = ""
        if isinstance(nodes, list) and node_id < len(nodes) and isinstance(nodes[node_id], dict):
            response_text = str(nodes[node_id].get("response") or "").strip()
        if response_text in exact_response_cache:
            judge_result = dict(exact_response_cache[response_text])
            exact_cache_hit_count += 1
            from_exact_cache = True
        else:
            judge_result, _, _ = env.sync_final_component_judge_text(
                question_text,
                response_text,
                gold_reference,
            )
            exact_response_cache[response_text] = dict(judge_result)
            from_exact_cache = False
        judge_failed = bool(judge_result.get("judge_failed", False))
        metrics = None if judge_failed else env._compute_agent_metrics(judge_result)
        total_claims = max(1, int(judge_result.get("total_claims", 0) or 0))
        is_hallucinating = None if judge_failed else (
            metrics["atomic_rate"] > 0.0 or bool(judge_result.get("is_refusal", False))
        )
        per_agent.append(
            {
                "agent_id": int(node_id),
                "response_text": response_text,
                "final_answer": response_text,
                "parsed_answer": env._parse_choice_answer(response_text),
                "judge_failed": judge_failed,
                "judge_failure_reason": judge_result.get("judge_failure_reason", ""),
                "evaluation_mode": str(judge_result.get("evaluation_mode", "llm_component_judge_v7")),
                "from_exact_response_cache": bool(from_exact_cache),
                "is_refusal": bool(judge_result.get("is_refusal", False)),
                "is_hallucinating": None if judge_failed else bool(is_hallucinating),
                "atomic_rate": None if judge_failed else float(metrics["atomic_rate"]),
                "swhi": None if judge_failed else float(metrics["swhi"]),
                "info_snr": None if judge_failed else float(metrics["info_snr"]),
                "minor_error_rate": None if judge_failed else float(int(judge_result.get("minor_errors", 0) or 0) / total_claims),
                "logical_flaw_rate": None if judge_failed else float(int(judge_result.get("logical_flaws", 0) or 0) / total_claims),
                "fabrication_rate": None if judge_failed else float(int(judge_result.get("fabrications", 0) or 0) / total_claims),
                "metrics": {
                    "total_claims": None if judge_failed else int(judge_result.get("total_claims", 0) or 0),
                    "correct_claims": None if judge_failed else int(judge_result.get("correct_claims", 0) or 0),
                    "minor_errors": None if judge_failed else int(judge_result.get("minor_errors", 0) or 0),
                    "logical_flaws": None if judge_failed else int(judge_result.get("logical_flaws", 0) or 0),
                    "fabrications": None if judge_failed else int(judge_result.get("fabrications", 0) or 0),
                },
            }
        )

    valid_agents = [item for item in per_agent if not item["judge_failed"]]
    n = len(valid_agents)
    mean_or_none = lambda key: (sum(float(item[key]) for item in valid_agents) / n) if n else None
    final_judge_audit = dict(getattr(env, "final_judge_audit_totals", {}) or {})
    final_judge_call_count = int(final_judge_audit.get("final_judge_call_count", 0) or 0)
    final_judge_capped_count = int(final_judge_audit.get("final_judge_capped_count", 0) or 0)
    final_judge_invalid_count = int(final_judge_audit.get("final_judge_invalid_count", 0) or 0)
    return {
        "per_agent": per_agent,
        "final_judge_failure_rate": (len(per_agent) - n) / max(1, len(per_agent)),
        "final_judge_call_count": final_judge_call_count,
        "final_judge_capped_count": final_judge_capped_count,
        "final_judge_capped_rate": final_judge_capped_count / max(1, final_judge_call_count),
        "final_judge_invalid_count": final_judge_invalid_count,
        "final_judge_invalid_rate": final_judge_invalid_count / max(1, final_judge_call_count),
        "final_judge_schema_retry_count": int(
            final_judge_audit.get("final_judge_schema_retry_count", 0) or 0
        ),
        "final_avg_node_real_hallu_rate": mean_or_none("is_hallucinating"),
        "final_avg_atomic_rate": mean_or_none("atomic_rate"),
        "final_avg_swhi": mean_or_none("swhi"),
        "final_avg_info_snr": mean_or_none("info_snr"),
        "final_refusal_rate": mean_or_none("is_refusal"),
        "final_avg_minor_error_rate": mean_or_none("minor_error_rate"),
        "final_avg_logical_flaw_rate": mean_or_none("logical_flaw_rate"),
        "final_avg_fabrication_rate": mean_or_none("fabrication_rate"),
        "final_total_claims": sum(int(item["metrics"]["total_claims"]) for item in valid_agents),
        "final_minor_errors": sum(int(item["metrics"]["minor_errors"]) for item in valid_agents),
        "final_logical_flaws": sum(int(item["metrics"]["logical_flaws"]) for item in valid_agents),
        "final_fabrications": sum(int(item["metrics"]["fabrications"]) for item in valid_agents),
        "offline_final_component_judge": True,
        "final_component_judge_prompt_version": env.FINAL_COMPONENT_JUDGE_VERSION,
        "final_component_scoring_basis": env.FINAL_COMPONENT_SCORING_BASIS,
        "final_component_deterministic_choice_agent_count": sum(
            1
            for item in per_agent
            if item.get("evaluation_mode") == env.FINAL_COMPONENT_CHOICE_ONLY_MODE
        ),
        "final_component_exact_cache_hit_count": int(exact_cache_hit_count),
    }


def _token_entry(prompt_tokens: int | float = 0, completion_tokens: int | float = 0) -> dict[str, int]:
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _build_token_usage_breakdown(
    token_usage: dict[str, Any],
    risk_judge_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate method judges, answer selection, and offline evaluation tokens."""

    audit = dict(risk_judge_audit or {})
    risk_claim = _token_entry(
        audit.get("risk_claim_judge_prompt_tokens", 0),
        audit.get("risk_claim_judge_completion_tokens", 0),
    )
    transmission = _token_entry(
        audit.get("transmission_judge_prompt_tokens", 0),
        audit.get("transmission_judge_completion_tokens", 0),
    )
    rep_judge = _token_entry(
        token_usage.get("rep_judge_prompt", 0),
        token_usage.get("rep_judge_completion", 0),
    )
    method_internal = _token_entry(
        risk_claim["prompt_tokens"] + transmission["prompt_tokens"] + rep_judge["prompt_tokens"],
        risk_claim["completion_tokens"] + transmission["completion_tokens"] + rep_judge["completion_tokens"],
    )
    offline_final = _token_entry(
        token_usage.get("final_judge_prompt", 0),
        token_usage.get("final_judge_completion", 0),
    )
    final_selector = _token_entry(
        token_usage.get("final_selector_prompt", 0),
        token_usage.get("final_selector_completion", 0),
    )
    return {
        "generation": _token_entry(
            token_usage.get("generation_prompt", 0),
            token_usage.get("generation_completion", 0),
        ),
        "method_online_risk_claim_judge": risk_claim,
        "method_transmission_judge": transmission,
        "method_rep_judge": rep_judge,
        "method_internal_judges_total": method_internal,
        "offline_final_evaluator": offline_final,
        "final_answer_selector": final_selector,
        "judge_total_recorded_compat": _token_entry(
            token_usage.get("judge_prompt", 0),
            token_usage.get("judge_completion", 0),
        ),
        "accounting_note": (
            "judge_total_recorded_compat is the legacy aggregate judge counter; do not add it to "
            "method_internal_judges_total, offline_final_evaluator, or final_answer_selector."
        ),
    }


def _api_component_entry(api_stats: dict[str, Any], locations: tuple[str, ...]) -> dict[str, float | int]:
    call_locations = dict(api_stats.get("call_locations") or {})
    attempt_locations = dict(api_stats.get("attempt_locations") or {})
    success_locations = dict(api_stats.get("success_locations") or {})
    failed_locations = dict(api_stats.get("failed_attempt_locations") or {})
    retry_locations = dict(api_stats.get("retry_locations") or {})
    failure_locations = dict(api_stats.get("failure_locations") or {})
    seconds_by_location = dict(api_stats.get("request_seconds_by_location") or {})
    return {
        "logical_call_count": sum(int(call_locations.get(name, 0) or 0) for name in locations),
        "request_attempt_count": sum(int(attempt_locations.get(name, 0) or 0) for name in locations),
        "successful_call_count": sum(int(success_locations.get(name, 0) or 0) for name in locations),
        "failed_attempt_count": sum(int(failed_locations.get(name, 0) or 0) for name in locations),
        "retry_count": sum(int(retry_locations.get(name, 0) or 0) for name in locations),
        "failure_count": sum(int(failure_locations.get(name, 0) or 0) for name in locations),
        "request_seconds_sum": sum(float(seconds_by_location.get(name, 0.0) or 0.0) for name in locations),
    }


def _build_api_request_breakdown(api_stats: dict[str, Any]) -> dict[str, Any]:
    """Separate API timing by semantic role for Ours."""

    groups = {
        "generation": ("generate_response",),
        "method_online_risk_claim_judge": ("risk_claim_extract",),
        "method_transmission_judge": (
            "risk_claim_inheritance",
            "risk_claim_inheritance_schema_retry",
        ),
        "method_rep_judge": ("rep_judge", "risk_review_judge", "verification_judge"),
        "offline_final_evaluator": ("offline_final_component_judge",),
        "final_answer_selector": ("final_answer_selector",),
    }
    used_locations = {name for locations in groups.values() for name in locations}
    all_locations = set(dict(api_stats.get("request_seconds_by_location") or {}))
    all_locations.update(dict(api_stats.get("call_locations") or {}))
    breakdown = {
        key: _api_component_entry(api_stats, locations)
        for key, locations in groups.items()
    }
    breakdown["other"] = _api_component_entry(api_stats, tuple(sorted(all_locations - used_locations)))
    breakdown["all_api_requests_recorded"] = {
        "logical_call_count": int(api_stats.get("logical_call_count", 0) or 0),
        "request_attempt_count": int(api_stats.get("request_attempt_count", 0) or 0),
        "successful_call_count": int(api_stats.get("successful_call_count", 0) or 0),
        "failed_attempt_count": int(api_stats.get("failed_attempt_count", 0) or 0),
        "retry_count": int(api_stats.get("retry_count", 0) or 0),
        "failure_count": int(api_stats.get("failure_count", 0) or 0),
        "request_seconds_sum": float(api_stats.get("request_seconds_sum", 0.0) or 0.0),
        "request_active_wall_seconds": float(api_stats.get("request_active_wall_seconds", 0.0) or 0.0),
        "api_wait_active_wall_seconds": float(api_stats.get("api_wait_active_wall_seconds", 0.0) or 0.0),
    }
    breakdown["accounting_note"] = (
        "method_* judge components are online method-internal calls; offline_final_evaluator is "
        "post-episode metric evaluation and must be reported separately."
    )
    return breakdown


def _has_unanimous_valid_answer(env: EvolutionEnvironment) -> bool:
    responses = env._snapshot_agent_outputs()
    choices = [env._parse_choice_answer(responses.get(agent_id, "")) for agent_id in env.agents]
    return bool(choices) and all(choice is not None and choice == choices[0] for choice in choices)


def run_single_episode(
    graph_dict: dict,
    config,
    *,
    calibration_summary: dict[str, Any] | None = None,
    enable_runtime_recalibration: bool = False,
    final_judge: bool = False,
    return_env: bool = False,
    fault_controller: Any | None = None,
) -> dict[str, Any]:
    env = EvolutionEnvironment(
        graph_dict=json.loads(json.dumps(graph_dict)),
        num_generations=config.egt.num_generations,
        config=config,
        fault_controller=fault_controller,
    )
    calibration = calibration_summary or apply_fixed_rule_parameter_calibration(config, env)
    history_stats = []
    env.agent_history = []
    env.tag_frequency_history = []
    env.strategy_stats_history = []
    env.action_history = []
    env.response_update_history = []
    env.graph_history = []
    env.transmission_history = []
    mutation_rate_scale = 1.0
    persona_effect_history: list[tuple[int, float]] = []
    previous_edges: set[tuple[int, int]] | None = None
    env.metric_initial_edges = set()
    env.metric_initial_risky_edges = set()
    env.metric_initial_benign_edges = set()
    env.metric_total_prune_count = 0
    env.metric_total_seek_attempts = 0
    env.metric_total_seek_successes = 0
    env.metric_total_self_reflect_count = 0
    env.metric_total_governance_attempts = 0
    env.metric_total_governance_effective_actions = 0
    initial_edge_sets_ready = False
    strategy_update_count = 0
    t_generations = int(config.egt.num_generations)
    termination_reason = "max_rounds"

    for t in range(t_generations):
        current_selection_alpha = config.get_selection_alpha(t, t_generations)
        base_mutation_rate = config.get_mutation_rate(t, t_generations)
        current_mutation_rate = _clamp(base_mutation_rate * mutation_rate_scale, 0.0, 1.0)
        current_exploration_noise = _annealed_noise(config.egt.exploration_noise, t, t_generations)
        risk_observer_calls_before_round = int(
            getattr(env, "risk_observer_invocation_count", 0) or 0
        )

        metric_summary = env.calculate_fitness(
            current_gen=t,
            verify_cost=config.egt.verify_cost,
            search_cost=config.egt.search_cost,
        )
        round_risk_state = env.latest_risk_state
        if not initial_edge_sets_ready:
            transmission_matrix = getattr(env, "latest_transmission_matrix", [])
            effective_beta = _current_edge_beta(env, config)
            (
                env.metric_initial_edges,
                env.metric_initial_risky_edges,
                env.metric_initial_benign_edges,
            ) = classify_edge_sets(
                env.graph_dict,
                env.agents,
                transmission_matrix,
                effective_beta,
            )
            initial_edge_sets_ready = True
        pre_governance_responses = env._snapshot_agent_outputs()
        env.execute_actions_and_update_graph()
        post_governance_responses = env._snapshot_agent_outputs()
        if fault_controller is not None:
            outgoing_source_ids = {
                int(edge[0])
                for edge in env.graph_dict.get("edges", []) or []
                if isinstance(edge, (list, tuple))
                and len(edge) >= 2
                and int(edge[0]) != int(edge[1])
            }
            fault_controller.finalize_round(
                round_index=t,
                outgoing_responses=post_governance_responses,
                visible_agent_ids=outgoing_source_ids,
                parse_answer=env._parse_choice_answer,
            )
        response_changed_agent_ids = sorted(
            node_id
            for node_id in set(pre_governance_responses) | set(post_governance_responses)
            if pre_governance_responses.get(node_id, "") != post_governance_responses.get(node_id, "")
        )
        post_governance_reobserved = bool(response_changed_agent_ids)
        post_governance_risk_state = (
            None
            if post_governance_reobserved
            else env.project_risk_state_to_current_graph(round_risk_state)
        )
        metric_summary = env.calculate_fitness(
            current_gen=t,
            verify_cost=config.egt.verify_cost,
            search_cost=config.egt.search_cost,
            refresh_pipeline=False,
            risk_state=post_governance_risk_state,
        )
        actual_risk_observer_passes = (
            int(getattr(env, "risk_observer_invocation_count", 0) or 0)
            - risk_observer_calls_before_round
        )
        latest_events = getattr(env, "latest_action_events", [])
        response_update_events = list(getattr(env, "latest_response_update_events", []) or [])
        action_metric_summary = summarize_action_events(latest_events)
        active_self_reflect = int(action_metric_summary.get("self_reflect_count", 0) or 0)

        fitness_list = [agent.fitness for agent in env.agents.values()]
        h_final_list = [agent.h_final for agent in env.agents.values()]
        rho_list = [getattr(agent, "rho", 0.0) for agent in env.agents.values()]
        pi_list = [agent.predicted_prob for agent in env.agents.values()]

        avg_fitness = sum(fitness_list) / len(fitness_list) if fitness_list else 0.0
        avg_h_final = sum(h_final_list) / len(h_final_list) if h_final_list else 0.0
        avg_pi = sum(pi_list) / len(pi_list) if pi_list else 0.0
        avg_rho = sum(rho_list) / len(rho_list) if rho_list else 0.0
        propagation_gain = avg_rho
        self_reflection_ratio = active_self_reflect / len(env.agents) if env.agents else 0.0
        edge_count = len(env.graph_dict.get("edges", []))
        current_edges = edge_set(env.graph_dict)
        transmission_matrix = getattr(env, "latest_transmission_matrix", [])
        effective_beta = _current_edge_beta(env, config)
        edge_quality_metrics = compute_edge_quality_metrics(
            env.graph_dict,
            env.agents,
            transmission_matrix,
            effective_beta,
        )
        topology_change_rate = compute_topology_change_rate(current_edges, previous_edges)
        topology_metrics = compute_topology_metrics(env.graph_dict)
        topology_risk_metrics = compute_topology_risk_metrics(
            env.graph_dict,
            env.agents,
            transmission_matrix,
            effective_beta,
            source_pi_by_edge=source_pi_by_edge_from_risk_state(env.latest_risk_state),
        )
        transmission_metrics = compute_transmission_metrics(getattr(env, "latest_edge_transmissions", []))
        strategy_ratios = env.get_strategy_ratios()

        stats = {
            "gen": t,
            "round": t + 1,
            "round_label": f"R{t + 1}",
            "avg_fitness": avg_fitness,
            "avg_pi": avg_pi,
            "avg_rho": avg_rho,
            "avg_h_final": avg_h_final,
            "avg_aggregate_risk": avg_h_final,
            "propagation_gain": propagation_gain,
            "self_reflection_ratio": self_reflection_ratio,
            "edge_count": edge_count,
            "effective_edge_ratio": edge_quality_metrics["effective_edge_ratio"],
            "isolated_node_ratio": edge_quality_metrics["isolated_node_ratio"],
            "edge_density": edge_quality_metrics["edge_density"],
            "avg_in_degree": edge_quality_metrics["avg_in_degree"],
            "avg_out_degree": edge_quality_metrics["avg_out_degree"],
            "topology_change_rate": topology_change_rate,
            "spectral_gap": topology_metrics["spectral_gap"],
            "avg_path_length": topology_metrics["avg_path_length"],
            "avg_clustering": topology_metrics["avg_clustering"],
            "edge_normalized_spectral_gap": topology_metrics["edge_normalized_spectral_gap"],
            "spectral_efficiency": topology_metrics["spectral_efficiency"],
            "topo_efficiency_legacy": topology_metrics["topo_efficiency_legacy"],
            "lcc_ratio": topology_metrics["lcc_ratio"],
            "degree_heterogeneity": topology_metrics["degree_heterogeneity"],
            "avg_edge_risk": topology_risk_metrics["avg_edge_risk"],
            "total_edge_risk": topology_risk_metrics["total_edge_risk"],
            "avg_adjusted_edge_risk": topology_risk_metrics["avg_adjusted_edge_risk"],
            "total_adjusted_edge_risk": topology_risk_metrics["total_adjusted_edge_risk"],
            "high_risk_edge_ratio": topology_risk_metrics["high_risk_edge_ratio"],
            "avg_edge_transmission": transmission_metrics["avg_edge_transmission"],
            "max_edge_transmission": transmission_metrics["max_edge_transmission"],
            "observed_transmission_coverage": transmission_metrics["observed_transmission_coverage"],
            "unknown_transmission_edge_count": transmission_metrics["unknown_transmission_edge_count"],
            "avg_infection_penalty": (
                sum(agent.infection_penalty for agent in env.agents.values()) / len(env.agents)
                if env.agents else 0.0
            ),
            "avg_node_real_hallu_rate": metric_summary["avg_node_real_hallu_rate"],
            "avg_atomic_rate": metric_summary["avg_atomic_rate"],
            "avg_swhi": metric_summary["avg_swhi"],
            "avg_info_snr": metric_summary["avg_info_snr"],
            "answer_choice_utility": float(metric_summary.get("avg_answer_choice_utility", 0.0) or 0.0),
            "consensus_answer": (getattr(env, "latest_consensus_analysis", {}) or {}).get("consensus_answer"),
            "consensus_risk": float(metric_summary.get("consensus_risk", 0.0) or 0.0),
            "consensus_ratio": float(metric_summary.get("consensus_ratio", 0.0) or 0.0),
            "consensus_breaker_triggered": float(metric_summary.get("consensus_breaker_triggered", 0.0) or 0.0),
            "selection_alpha_used": float(current_selection_alpha),
            "selection_alpha_start": float(config.egt.selection_alpha_start),
            "selection_alpha_end": float(config.egt.selection_alpha_end),
            "edge_beta_used": effective_beta,
            "edge_beta_raw": float(getattr(env, "latest_edge_beta_raw", 0.0) or 0.0),
            "edge_beta_capped": bool(getattr(env, "latest_edge_beta_capped", False)),
            "edge_beta_fallback_applied": bool(
                getattr(env, "latest_edge_beta_fallback_applied", False)
            ),
            "edge_risk_ratio_q85": float(getattr(env, "latest_edge_risk_ratio_q85", 0.0) or 0.0),
            "avg_edge_risk_ratio": float(getattr(env, "latest_avg_edge_risk_ratio", 0.0) or 0.0),
            "max_edge_risk_ratio": float(getattr(env, "latest_max_edge_risk_ratio", 0.0) or 0.0),
            "risk_state_timing": (
                "post_governance_reobserved" if post_governance_reobserved else "pre_governance_observed"
            ),
            "post_governance_risk_mode": (
                "reobserved_after_response_change"
                if post_governance_reobserved
                else "projected_from_same_round_observation"
            ),
            "post_governance_reobserved": post_governance_reobserved,
            "response_changed_agent_ids": response_changed_agent_ids,
            "online_risk_observer_passes_this_round": int(actual_risk_observer_passes),
            "risk_observer_calls_cumulative": int(
                getattr(env, "risk_observer_invocation_count", 0) or 0
            ),
            "response_update_accept_count": sum(
                1 for event in response_update_events if bool(event.get("candidate_accepted", False))
            ),
            "invalid_candidate_preserved_count": sum(
                1 for event in response_update_events if bool(event.get("invalid_candidate_preserved", False))
            ),
            "generation_failure_count": sum(
                1 for event in response_update_events if bool(event.get("generation_failed", False))
            ),
        }
        latest_risk_details = dict(getattr(env, "latest_prediction_details", {}).get("details", {}) or {})
        for key in (
            "risk_judge_prompt_tokens",
            "risk_judge_completion_tokens",
            "risk_judge_capped_count",
            "risk_judge_capped_rate",
            "risk_judge_invalid_count",
            "risk_judge_invalid_rate",
            "risk_judge_schema_retry_count",
            "risk_judge_call_count",
            "risk_claim_judge_prompt_tokens",
            "risk_claim_judge_completion_tokens",
            "risk_claim_judge_call_count",
            "transmission_judge_prompt_tokens",
            "transmission_judge_completion_tokens",
            "transmission_judge_call_count",
            "missing_before_response_count",
            "current_claim_cache_hit_count",
        ):
            stats[key] = latest_risk_details.get(key, 0)
        stats.update(strategy_ratios)
        stats.update(action_metric_summary)
        history_stats.append(stats)

        agent_snapshot = {"gen": t, "round": t + 1, "round_label": f"R{t + 1}", "agents": []}
        tag_counts = {}
        rep_count = 0
        seek_count = 0
        for agent_id, agent in env.agents.items():
            agent_snapshot["agents"].append(
                {
                    "agent_id": agent_id,
                    "tags": sorted(agent.tags),
                    "initial_phenotype_tags": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("initial_phenotype_tags", []),
                    "initial_phenotype_tags_source": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("initial_phenotype_tags_source", ""),
                    "role_text": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("role_text", ""),
                    "raw_role_text": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("raw_role_text", ""),
                    "role_adapter_id": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("role_adapter_id", ""),
                    "role_adapter_text": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("role_adapter_text", ""),
                    "dynamic_role_wrapper": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("dynamic_role_wrapper", ""),
                    "risk_status": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("risk_status", ""),
                    "phenotype_tags_block": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("phenotype_tags_block", ""),
                    "composed_role_prompt": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("composed_role_prompt", ""),
                    "fitness": agent.fitness,
                    "pi": getattr(agent, "pi", None),
                    "predicted_prob": getattr(agent, "predicted_prob", None),
                    "rho": getattr(agent, "rho", None),
                    "h_pre": agent.h_pre,
                    "h_final": agent.h_final,
                    "intrinsic_utility": agent.intrinsic_utility,
                    "infection_penalty": agent.infection_penalty,
                    "a_self": agent.a_self,
                    "a_rep": agent.a_rep,
                    "a_seek": agent.a_seek,
                    "strategy_id": int(getattr(agent, "strategy_id", -1)),
                    "initial_strategy_source_id": int(
                        getattr(agent, "initial_strategy_source_id", -1)
                    ),
                    "initial_strategy_source": list(
                        getattr(agent, "initial_strategy_source", ())
                    ),
                    "initial_projected_strategy": list(
                        getattr(agent, "initial_projected_strategy", ())
                    ),
                    "use_reflection": agent.use_reflection,
                    "rep_success_count": agent.rep_success_count,
                    "reported_count": agent.reported_count,
                    "real_hallucination_rate": agent.real_hallucination_rate,
                    "metric_atomic": agent.metric_atomic,
                    "metric_swhi": agent.metric_swhi,
                    "metric_snr": agent.metric_snr,
                    "answer_choice_utility": getattr(agent, "answer_choice_utility", 0.0),
                    "consensus_risk_exposure": getattr(agent, "consensus_risk_exposure", 0.0),
                }
            )
            rep_count += 1 if agent.a_rep == 1 else 0
            seek_count += 1 if agent.a_seek == 1 else 0
            for tag in agent.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        env.agent_history.append(agent_snapshot)
        env.tag_frequency_history.append({"gen": t, "tag_counts": tag_counts})
        env.transmission_history.append(
            {
                "gen": t,
                "round": t + 1,
                "round_label": f"R{t + 1}",
                "edges": normalize_edge_transmission_records(getattr(env, "latest_edge_transmissions", [])),
            }
        )
        env.strategy_stats_history.append(
            {
                "gen": t,
                "round": t + 1,
                "round_label": f"R{t + 1}",
                "a_self_ratio": self_reflection_ratio,
                "a_rep_ratio": rep_count / len(env.agents) if env.agents else 0.0,
                "a_seek_ratio": seek_count / len(env.agents) if env.agents else 0.0,
                "avg_rep_success_count": (
                    sum(agent.rep_success_count for agent in env.agents.values()) / len(env.agents)
                    if env.agents else 0.0
                ),
                "avg_reported_count": (
                    sum(agent.reported_count for agent in env.agents.values()) / len(env.agents)
                    if env.agents else 0.0
                ),
                "answer_choice_utility": float(metric_summary.get("avg_answer_choice_utility", 0.0) or 0.0),
                "consensus_risk": float(metric_summary.get("consensus_risk", 0.0) or 0.0),
                "consensus_ratio": float(metric_summary.get("consensus_ratio", 0.0) or 0.0),
                "consensus_breaker_triggered": float(metric_summary.get("consensus_breaker_triggered", 0.0) or 0.0),
                "strategy_ratios": strategy_ratios,
            }
        )

        if t % max(1, t_generations // 10) == 0 or t == t_generations - 1:
            print(
                f"[Generation {t:03d}] Atomic={_format_metric(metric_summary['avg_atomic_rate'])} | "
                f"SWHI={_format_metric(metric_summary['avg_swhi'])} | "
                f"InfoSNR={_format_metric(metric_summary['avg_info_snr'])} | "
                f"Fitness={avg_fitness:.3f} | Edges={edge_count} | "
                f"SelectionAlpha={current_selection_alpha:.3f} | EdgeBeta={effective_beta:.3f} | Mutation={current_mutation_rate:.3f} | "
                f"Noise={current_exploration_noise:.3f}"
            )

        observed_tag_replacements = 0
        for agent_events in latest_events:
            for ev in agent_events.get("events", []):
                if ev.get("type") != "self_reflect" or not bool(ev.get("accepted", False)):
                    continue
                old_tags = set(ev.get("old_tags", []) or [])
                new_tags = set(ev.get("new_tags", []) or [])
                replaced = len(old_tags - new_tags)
                if replaced > 0:
                    observed_tag_replacements += replaced
                base_u = float(ev.get("baseline_structural_fitness", 0.0) or 0.0)
                cand_u = float(ev.get("candidate_structural_fitness", 0.0) or 0.0)
                delta_u_abs = abs(cand_u - base_u)
                if replaced > 0 and delta_u_abs >= 0.0:
                    persona_effect_history.append((replaced, delta_u_abs))

        env.metric_total_prune_count += int(action_metric_summary["prune_count"])
        env.metric_total_seek_attempts += int(action_metric_summary["seek_attempt_count"])
        env.metric_total_seek_successes += int(action_metric_summary["seek_success_count"])
        env.metric_total_self_reflect_count += int(action_metric_summary["self_reflect_count"])
        env.metric_total_governance_attempts += int(action_metric_summary["governance_attempt_count"])
        env.metric_total_governance_effective_actions += int(action_metric_summary["governance_effective_count"])

        avg_tag_count = (
            sum(len(agent.tags) for agent in env.agents.values()) / max(1, len(env.agents))
            if env.agents else 1.0
        )
        avg_tag_count = max(1.0, float(avg_tag_count))
        expected_replacements = active_self_reflect * current_mutation_rate * avg_tag_count
        if bool(enable_runtime_recalibration) and active_self_reflect > 0 and observed_tag_replacements > 0:
            correction = expected_replacements / max(1e-6, float(observed_tag_replacements))
            mutation_rate_scale = _clamp(correction, 0.5, 2.0)

        if bool(enable_runtime_recalibration) and len(persona_effect_history) >= 5:
            deltas = [d for _, d in persona_effect_history]
            tau_min = _quantile(deltas, 0.25)
            tau_max = _quantile(deltas, 0.75)
            grouped: dict[int, list[float]] = {}
            for repl_count, dval in persona_effect_history:
                grouped.setdefault(int(repl_count), []).append(float(dval))
            scored = []
            for repl_count, vals in grouped.items():
                if not vals:
                    continue
                hit_prob = sum(1 for v in vals if tau_min <= v <= tau_max) / len(vals)
                scored.append((hit_prob, repl_count))
            if scored:
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_replacements = max(1, int(scored[0][1]))
                config.egt.persona_mutation_rate = _clamp(best_replacements / avg_tag_count, 0.01, 1.0)

        env.action_history.append(
            {"gen": t, "round": t + 1, "round_label": f"R{t + 1}", "agent_events": latest_events}
        )
        env.response_update_history.append(
            {"gen": t, "round": t + 1, "round_label": f"R{t + 1}", "events": response_update_events}
        )
        env.graph_history.append(
            {
                "gen": t,
                "round": t + 1,
                "round_label": f"R{t + 1}",
                "edges": [
                    [int(edge[0]), int(edge[1])]
                    for edge in env.graph_dict.get("edges", [])
                    if len(edge) >= 2
                ],
                "nodes": [
                    {
                        "agent_id": int(agent_id),
                        "initial_phenotype_tags": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("initial_phenotype_tags", []),
                        "initial_phenotype_tags_source": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("initial_phenotype_tags_source", ""),
                        "role_text": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("role_text", ""),
                        "raw_role_text": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("raw_role_text", ""),
                        "role_adapter_id": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("role_adapter_id", ""),
                        "role_adapter_text": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("role_adapter_text", ""),
                        "dynamic_role_wrapper": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("dynamic_role_wrapper", ""),
                        "risk_status": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("risk_status", ""),
                        "phenotype_tags_block": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("phenotype_tags_block", ""),
                        "composed_role_prompt": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("composed_role_prompt", ""),
                        "response": env.graph_dict.get("nodes", [{} for _ in env.agents])[agent_id].get("response", ""),
                    }
                    for agent_id in sorted(env.agents.keys())
                ],
            }
        )
        previous_edges = edge_set(env.graph_dict)
        functional_spec = get_functional_ablation_spec(
            getattr(config.egt, "functional_ablation_mode", "full")
        )
        if functional_spec.strategy_evolution_enabled:
            env.update_strategies_fermi(
                selection_alpha=current_selection_alpha,
                mutation_rate=current_mutation_rate,
                exploration_noise=current_exploration_noise,
            )
            strategy_update_count += 1
        else:
            # Restore actions temporarily gated in gen-0 without changing strategy ids.
            for agent in env.agents.values():
                agent.apply_strategy(agent.strategy_id)

        # Keep both perturbation protocols on their requested schedule.
        if fault_controller is None and not config.perturbation.enabled:
            if _has_unanimous_valid_answer(env):
                termination_reason = "unanimous_consensus"
                break

    final_stats = dict(history_stats[-1]) if history_stats else {}
    final_stats["termination_reason"] = termination_reason
    final_answer_eval = _build_final_answer_eval(env, config)
    if final_judge:
        final_eval = _evaluate_final_layer_with_judge(env, config)
    else:
        final_eval = {
            "per_agent": [],
            "final_judge_failure_rate": 0.0,
            "final_judge_call_count": 0,
            "final_judge_capped_count": 0,
            "final_judge_capped_rate": 0.0,
            "final_judge_invalid_count": 0,
            "final_judge_invalid_rate": 0.0,
            "final_judge_schema_retry_count": 0,
            "final_avg_node_real_hallu_rate": 0.0,
            "final_avg_atomic_rate": 0.0,
            "final_avg_swhi": 0.0,
            "final_avg_info_snr": 0.0,
            "final_refusal_rate": 0.0,
        }
    env.latest_final_eval = final_eval
    env.latest_final_answer_eval = final_answer_eval
    final_stats.update({k: v for k, v in final_eval.items() if k != "per_agent"})
    final_stats.update(
        {
            "gold_answer": final_answer_eval.get("gold_answer"),
            "question_final_answer": final_answer_eval.get("question_final_answer"),
            "question_final_parsed_answer": final_answer_eval.get("question_final_parsed_answer"),
            "majority_vote_answer": final_answer_eval.get("majority_vote_answer"),
            "majority_vote_parsed_answer": final_answer_eval.get("majority_vote_parsed_answer"),
            "selector_answer": final_answer_eval.get("selector_answer"),
            "selector_accuracy": final_answer_eval.get("selector_accuracy"),
            "selector_status": final_answer_eval.get("selector_status"),
            "lowest_risk_agent_answer": final_answer_eval.get("lowest_risk_agent_answer"),
            "lowest_risk_agent_parsed_answer": final_answer_eval.get("lowest_risk_agent_parsed_answer"),
            "final_selection_method": final_answer_eval.get("final_selection_method"),
            "final_selection_uses_risk": final_answer_eval.get("final_selection_uses_risk"),
            "valid_final_answer_count": final_answer_eval.get("valid_final_answer_count"),
            "final_format_retry_triggered": final_answer_eval.get("final_format_retry_triggered"),
            "final_format_retry_succeeded": final_answer_eval.get("final_format_retry_succeeded"),
            "question_final_accuracy": final_answer_eval.get("question_final_accuracy"),
            "majority_vote_accuracy": final_answer_eval.get("majority_vote_accuracy"),
            "lowest_risk_agent_accuracy": final_answer_eval.get("lowest_risk_agent_accuracy"),
            "agent_answer_accuracy": final_answer_eval.get("agent_answer_accuracy"),
            "agent_answer_accuracy_valid_only": final_answer_eval.get("agent_answer_accuracy_valid_only"),
            "parse_fail_rate": final_answer_eval.get("parse_fail_rate"),
            "final_agent_answer_accuracy": final_answer_eval.get("agent_answer_accuracy"),
            "final_agent_answer_accuracy_valid_only": final_answer_eval.get("agent_answer_accuracy_valid_only"),
            "final_answer_parse_fail_rate": final_answer_eval.get("parse_fail_rate"),
            "accuracy": final_answer_eval.get("question_final_accuracy"),
            "final_answer_accuracy": final_answer_eval.get("question_final_accuracy"),
        }
    )
    token_usage = dict(getattr(env, "token_usage", {}))
    risk_judge_audit = _build_cumulative_risk_judge_audit(env)
    api_request_stats = env.get_api_request_stats()
    token_usage_by_model = _build_model_token_stats(token_usage, config)
    final_edges = edge_set(env.graph_dict)
    final_transmission_matrix = getattr(env, "latest_transmission_matrix", [])
    final_beta = _current_edge_beta(env, config)
    final_edge_quality = compute_edge_quality_metrics(
        env.graph_dict,
        env.agents,
        final_transmission_matrix,
        final_beta,
    )
    final_mechanism_metrics = compute_final_edge_mechanism_metrics(
        initial_edges=getattr(env, "metric_initial_edges", set()),
        initial_risky_edges=getattr(env, "metric_initial_risky_edges", set()),
        initial_benign_edges=getattr(env, "metric_initial_benign_edges", set()),
        final_edges=final_edges,
        agents=env.agents,
        graph_dict=env.graph_dict,
        transmission_matrix=final_transmission_matrix,
        beta=final_beta,
    )
    rho_area = sum(float(row.get("avg_rho", row.get("avg_h_final", 0.0)) or 0.0) for row in history_stats)
    aggregate_risk_area = sum(float(row.get("avg_h_final", 0.0) or 0.0) for row in history_stats)
    utility_area = sum(float(row.get("avg_fitness", 0.0) or 0.0) for row in history_stats)
    edge_risk_area = sum(float(row.get("avg_edge_risk", 0.0) or 0.0) for row in history_stats)
    adjusted_edge_risk_area = sum(float(row.get("avg_adjusted_edge_risk", 0.0) or 0.0) for row in history_stats)
    first_rho = float(history_stats[0].get("avg_rho", history_stats[0].get("avg_h_final", 0.0)) or 0.0) if history_stats else 0.0
    last_rho = float(final_stats.get("avg_rho", final_stats.get("avg_h_final", 0.0)) or 0.0)
    rho_reduction = first_rho - last_rho
    first_h = float(history_stats[0].get("avg_h_final", 0.0) or 0.0) if history_stats else 0.0
    last_h = float(final_stats.get("avg_h_final", 0.0) or 0.0)
    aggregate_risk_reduction = first_h - last_h
    total_governance_attempts = int(getattr(env, "metric_total_governance_attempts", 0))
    total_governance_effective_actions = int(getattr(env, "metric_total_governance_effective_actions", 0))
    if rho_reduction > 1e-8 and total_governance_effective_actions > 0:
        action_efficiency = rho_reduction / total_governance_effective_actions
    else:
        action_efficiency = None
    latest_risk_details = dict(getattr(env, "latest_prediction_details", {}).get("details", {}) or {})
    final_stats.update(
        {
            "post_action_edge_count": len(final_edges),
            "collapse_flag": 1.0 if len(final_edges) == 0 else 0.0,
            "rho_area": rho_area,
            "aggregate_risk_area": aggregate_risk_area,
            "utility_area": utility_area,
            "edge_risk_area": edge_risk_area,
            "adjusted_edge_risk_area": adjusted_edge_risk_area,
            "rho_auc_legacy": rho_area,
            "risk_auc_legacy": aggregate_risk_area,
            "utility_auc_legacy": utility_area,
            "edge_risk_auc_legacy": edge_risk_area,
            "rho_reduction": rho_reduction,
            "risk_reduction": aggregate_risk_reduction,
            "action_efficiency": action_efficiency,
            "total_prune_count": int(getattr(env, "metric_total_prune_count", 0)),
            "total_seek_attempts": int(getattr(env, "metric_total_seek_attempts", 0)),
            "total_seek_successes": int(getattr(env, "metric_total_seek_successes", 0)),
            "total_self_reflect_count": int(getattr(env, "metric_total_self_reflect_count", 0)),
            "total_governance_attempts": total_governance_attempts,
            "total_governance_effective_actions": total_governance_effective_actions,
            "total_generation_failure_count": sum(
                int(row.get("generation_failure_count", 0) or 0)
                for row in history_stats
            ),
            "overall_rewiring_success_rate": (
                int(getattr(env, "metric_total_seek_successes", 0))
                / (int(getattr(env, "metric_total_seek_attempts", 0)) + 1e-8)
            ),
            "selection_alpha_used": float(history_stats[-1].get("selection_alpha_used", 0.0) if history_stats else 0.0),
            "selection_alpha_start": float(config.egt.selection_alpha_start),
            "selection_alpha_end": float(config.egt.selection_alpha_end),
            "edge_beta_used": final_beta,
            "edge_beta_raw": float(getattr(env, "latest_edge_beta_raw", 0.0) or 0.0),
            "edge_beta_capped": bool(getattr(env, "latest_edge_beta_capped", False)),
            "edge_risk_ratio_q85": float(getattr(env, "latest_edge_risk_ratio_q85", 0.0) or 0.0),
            "avg_edge_risk_ratio": float(getattr(env, "latest_avg_edge_risk_ratio", 0.0) or 0.0),
            "max_edge_risk_ratio": float(getattr(env, "latest_max_edge_risk_ratio", 0.0) or 0.0),
            "answer_choice_utility": float((getattr(env, "latest_answer_choice_analysis", {}) or {}).get("avg_answer_choice_utility", 0.0) or 0.0),
            "consensus_answer": (getattr(env, "latest_consensus_analysis", {}) or {}).get("consensus_answer"),
            "consensus_ratio": float((getattr(env, "latest_consensus_analysis", {}) or {}).get("consensus_ratio", 0.0) or 0.0),
            "consensus_risk": float((getattr(env, "latest_consensus_analysis", {}) or {}).get("consensus_risk", 0.0) or 0.0),
            "consensus_breaker_triggered": bool((getattr(env, "latest_consensus_analysis", {}) or {}).get("consensus_breaker_triggered", False)),
            **final_edge_quality,
            **final_mechanism_metrics,
        }
    )
    triggered_rows = [row for row in history_stats if bool(row.get("consensus_breaker_triggered", 0.0))]
    triggered_consensus_answer = triggered_rows[0].get("consensus_answer") if triggered_rows else None
    max_triggered_consensus_risk = max((float(row.get("consensus_risk", 0.0) or 0.0) for row in triggered_rows), default=0.0)
    final_consensus = dict(getattr(env, "latest_consensus_analysis", {}) or {})
    final_stats["consensus_breaker_success"] = bool(
        triggered_rows
        and (
            float(final_consensus.get("consensus_risk", 0.0) or 0.0) + 1e-8 < max_triggered_consensus_risk
            or final_consensus.get("consensus_answer") != triggered_consensus_answer
        )
    )
    final_stats.update(risk_judge_audit)
    final_stats.update(
        {
            "online_risk_judge_tokens": int(token_usage.get("online_risk_judge_prompt", 0) or 0)
            + int(token_usage.get("online_risk_judge_completion", 0) or 0),
            "final_offline_judge_tokens": int(token_usage.get("final_judge_prompt", 0) or 0)
            + int(token_usage.get("final_judge_completion", 0) or 0),
        }
    )

    result = {
        "meta": {
            "risk_estimator": "llm_judge_atomic_observer",
            "risk_estimator_mode": str(getattr(config.egt, "risk_estimator_mode", "judge_observer")),
            "risk_judge_used": True,
            "final_scoring_basis": "llm_judge_atomic_pi_rho_T_risk_fitness",
            "invalid_response_fallback": bool(
                getattr(config.egt, "enable_invalid_response_fallback", True)
            ),
            "response_update_policy": (
                "invalid_only_preserve_previous_valid_v1"
                if bool(getattr(config.egt, "enable_invalid_response_fallback", True))
                else "unconditional_candidate_commit_legacy"
            ),
            "online_risk_prompt_protocol_version": (
                "reference_free_argument_compact_split_context_v4"
                if bool(getattr(config.egt, "deduplicate_online_risk_prompt_context", True))
                else "duplicated_context_v1"
            ),
            "risk_judge_prompt_version": env.RISK_JUDGE_PROMPT_VERSION,
            "transmission_judge_prompt_version": env.TRANSMISSION_JUDGE_PROMPT_VERSION,
            "risk_context_has_peer_messages": False,
            "online_judge_has_gold": False,
            "transmission_uses_before_after": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).semantic_transmission_enabled,
            "transmission_sender_risk_visible": False,
            "transmission_judge_used": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).semantic_transmission_enabled,
            "reuse_unchanged_claim_observations": bool(
                getattr(config.egt, "reuse_unchanged_claim_observations", True)
            ),
            "online_risk_observer_oracle_free": bool(
                getattr(config.egt, "online_risk_observer_oracle_free", True)
            ),
            "online_risk_observer_uses_gold": False,
            "offline_final_evaluator_uses_gold": True,
            "offline_final_component_judge": bool(final_judge),
            "final_component_judge_prompt_version": (
                env.FINAL_COMPONENT_JUDGE_VERSION if final_judge else None
            ),
            "final_component_scoring_basis": (
                env.FINAL_COMPONENT_SCORING_BASIS if final_judge else None
            ),
            "method_protocol_version": env.METHOD_PROTOCOL_VERSION,
            "peer_message_prompt_version": env.PEER_MESSAGE_PROMPT_VERSION,
            "transmission_timing": "sender_t_minus_1_to_receiver_t",
            "unknown_transmission_policy": (
                "realized_invalid_or_unobserved_T0;prospective_seek_screening_T1"
            ),
            "transmission_claim_risk_threshold": float(
                config.egt.transmission_claim_risk_threshold
            ),
            "initial_state_risk_observer_passes": 1,
            "initial_state_message_policy": "raw_output_only",
            "initial_state_transmission_identifiable": False,
            "initial_state_rho_mode": "not_inferred_before_peer_message_consumption",
            "pre_governance_risk_observer_passes_per_round": 1,
            "post_governance_risk_observer_policy": "rerun_only_if_agent_response_changed",
            "max_main_risk_observer_passes_per_round": max(
                (
                    int(row.get("online_risk_observer_passes_this_round", 0) or 0)
                    for row in history_stats
                ),
                default=0,
            ),
            "risk_observer_pass_count_semantics": (
                "actual_full_state_estimator_invocations_including_self_candidates"
            ),
            "risk_observer_invocation_count": int(
                getattr(env, "risk_observer_invocation_count", 0) or 0
            ),
            "response_refresh_interval": int(config.egt.response_refresh_interval),
            "edge_beta_definition": (
                "configured_beta_if_no_valid_ratios;"
                "1_if_q85_le_eps_else_min(q85_edge_risk_ratio,cap)"
            ),
            "selection_alpha_schedule": "linear",
            "strategy_mutation_operator": "single_bit_flip",
            "fermi_candidate_pool": "all_other_agents",
            "functional_ablation_mode": str(
                getattr(config.egt, "functional_ablation_mode", "full")
            ),
            "functional_ablation_stage": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).cumulative_stage,
            "self_reflection_mode": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).self_reflection_mode,
            "strategy_evolution_enabled": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).strategy_evolution_enabled,
            "topology_governance_enabled": get_functional_ablation_spec(
                getattr(config.egt, "functional_ablation_mode", "full")
            ).topology_governance_enabled,
            "transmission_estimator_mode": (
                "semantic_judge"
                if get_functional_ablation_spec(
                    getattr(config.egt, "functional_ablation_mode", "full")
                ).semantic_transmission_enabled
                else "structural_edge"
            ),
            "strategy_update_count": int(strategy_update_count),
            "online_consensus_uses_injection_wrong_label": False,
            "actual_response_generation_rounds": int(getattr(env, "response_generation_round_count", 0)),
            "round_indexing": "gen_0_based_round_1_based",
            "risk_area_scope": "generated_rounds_only_excludes_injected_round_0",
            "final_answer_selection": "majority_vote_and_gold_blind_deepseek_selector_v1",
            "final_answer_selection_uses_risk": False,
            "final_format_retry_enabled": False,
            "self_reflection_generation_context": _self_reflection_generation_context(config),
        },
        "calibration_summary": calibration,
        "termination_reason": termination_reason,
        "final_summary": final_stats,
        "history_stats": history_stats,
        "history_tail": history_stats[-5:] if len(history_stats) >= 5 else history_stats,
        "agent_history": list(getattr(env, "agent_history", [])),
        "action_history": list(getattr(env, "action_history", [])),
        "response_update_history": list(getattr(env, "response_update_history", [])),
        "graph_history": list(getattr(env, "graph_history", [])),
        "transmission_history": list(getattr(env, "transmission_history", [])),
        "initial_risk_state": (
            env.initial_risk_state.to_legacy_details(size=len(env.agents))
            if env.initial_risk_state is not None
            else None
        ),
        "strategy_stats_history": list(getattr(env, "strategy_stats_history", [])),
        "final_eval": final_eval,
        "final_answer_eval": final_answer_eval,
        "answer_choice_analysis": dict(getattr(env, "latest_answer_choice_analysis", {}) or {}),
        "consensus_analysis": final_consensus,
        "online_prompt_samples": list(getattr(env, "latest_online_prompt_samples", []) or []),
        "token_usage": token_usage,
        "token_usage_breakdown": _build_token_usage_breakdown(token_usage, risk_judge_audit),
        "token_usage_by_model": token_usage_by_model,
        "api_request_stats": api_request_stats,
        "api_request_breakdown": _build_api_request_breakdown(api_request_stats),
        "rep_judge_cache_hits": int(getattr(env, "rep_judge_cache_hits", 0)),
        "rep_judge_cache_size": len(getattr(env, "rep_judge_cache", {})),
        "risk_judge_audit": risk_judge_audit,
        "gold_answer": final_answer_eval.get("gold_answer"),
        "question_final_answer": final_answer_eval.get("question_final_answer"),
        "question_final_parsed_answer": final_answer_eval.get("question_final_parsed_answer"),
        "majority_vote_answer": final_answer_eval.get("majority_vote_answer"),
        "majority_vote_parsed_answer": final_answer_eval.get("majority_vote_parsed_answer"),
        "selector_answer": final_answer_eval.get("selector_answer"),
        "selector_parsed_answer": final_answer_eval.get("selector_parsed_answer"),
        "selector_accuracy": final_answer_eval.get("selector_accuracy"),
        "selector_status": final_answer_eval.get("selector_status"),
        "selector_model": final_answer_eval.get("selector_model"),
        "selector_error": final_answer_eval.get("selector_error"),
        "lowest_risk_agent_answer": final_answer_eval.get("lowest_risk_agent_answer"),
        "lowest_risk_agent_parsed_answer": final_answer_eval.get("lowest_risk_agent_parsed_answer"),
        "final_selection_method": final_answer_eval.get("final_selection_method"),
        "final_selection_uses_risk": final_answer_eval.get("final_selection_uses_risk"),
        "valid_final_answer_count": final_answer_eval.get("valid_final_answer_count"),
        "final_format_retry_triggered": final_answer_eval.get("final_format_retry_triggered"),
        "final_format_retry_succeeded": final_answer_eval.get("final_format_retry_succeeded"),
        "per_agent_final_answers": final_answer_eval.get("per_agent_final_answers", []),
        "question_final_accuracy": final_answer_eval.get("question_final_accuracy"),
        "majority_vote_accuracy": final_answer_eval.get("majority_vote_accuracy"),
        "lowest_risk_agent_accuracy": final_answer_eval.get("lowest_risk_agent_accuracy"),
        "agent_answer_accuracy": final_answer_eval.get("agent_answer_accuracy"),
        "agent_answer_accuracy_valid_only": final_answer_eval.get("agent_answer_accuracy_valid_only"),
        "parse_fail_rate": final_answer_eval.get("parse_fail_rate"),
        "answer_choice_utility": float((getattr(env, "latest_answer_choice_analysis", {}) or {}).get("avg_answer_choice_utility", 0.0) or 0.0),
        "consensus_answer": final_consensus.get("consensus_answer"),
        "consensus_ratio": float(final_consensus.get("consensus_ratio", 0.0) or 0.0),
        "consensus_risk": float(final_consensus.get("consensus_risk", 0.0) or 0.0),
        "consensus_breaker_triggered": bool(final_consensus.get("consensus_breaker_triggered", False)),
        "consensus_breaker_success": bool(final_stats.get("consensus_breaker_success", False)),
    }
    if fault_controller is not None:
        from .two_window import result_block as two_window_result_block

        result.update(two_window_result_block(fault_controller))
    if return_env:
        result["env"] = env
    else:
        env.close_debug_file()
    return result


def main(argv: list[str] | None = None):
    config = build_default_config()
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge atomic observer evolutionary experiment.")
    parser.add_argument("--seed", type=int, default=config.egt.seed, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--input-graph",
        type=Path,
        default=config.paths.input_graph_path,
        help="Input graph JSONL; required if the default generated data file is absent.",
    )
    parser.add_argument("--line-idx", type=int, default=0, help="Zero-based JSONL row to run.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write this episode's result to a specific JSON file.")
    parser.add_argument("--output-figure", type=Path, default=None, help="Write this episode's figure to a specific PNG file.")
    parser.add_argument("--generation-model", default=None, help="Model preset for agent responses.")
    parser.add_argument("--prompt-model", default=None, help="Model preset for prompt generation; defaults to generation model when overridden.")
    parser.add_argument("--judge-model", default=None, help="Model preset for online and final judges.")
    parser.add_argument("--generation-key-env", default=None, help="Environment variable holding the generation API key.")
    parser.add_argument("--prompt-key-env", default=None, help="Environment variable holding the prompt API key.")
    parser.add_argument("--judge-key-env", default=None, help="Environment variable holding the judge API key.")
    parser.add_argument(
        "--functional-ablation-mode",
        choices=FUNCTIONAL_ABLATION_CHOICES,
        default="full",
        help="Select the full method or a functional ablation.",
    )
    parser.add_argument(
        "--strategy-mode",
        choices=("full", "no_self"),
        default="full",
        help="Disable one strategy action for an action-level ablation.",
    )
    parser.add_argument("--enable-peer-context", action="store_true", help="Pass neighbor responses into generation prompts.")
    parser.add_argument("--peer-context-max-chars", type=int, default=config.egt.peer_context_max_chars)
    parser.add_argument(
        "--prompt-style",
        choices=["default", "aligned"],
        default=str(config.egt.prompt_style),
    )
    parser.add_argument("--disable-gene-role-prompts", action="store_true")
    parser.add_argument(
        "--risk-estimator-mode",
        type=str,
        default=str(config.egt.risk_estimator_mode),
        choices=["judge_observer"],
        help="Main experiment path uses the LLM-as-Judge atomic observer.",
    )
    parser.add_argument("--beta-mode", type=str, default=None, choices=["fixed", "adaptive"])
    parser.add_argument("--fitness-penalty-beta", type=float, default=None)
    parser.add_argument(
        "--adaptive-beta-quantile",
        "--beta-quantile",
        dest="adaptive_beta_quantile",
        type=float,
        default=None,
        help="Quantile q used by adaptive beta normalization.",
    )
    parser.add_argument("--disable-cache", action="store_true", help="Disable response and judge caches for audit runs.")
    parser.add_argument(
        "--final-judge", action=argparse.BooleanOptionalAction, default=True,
        help="Run the paper's gold-aware offline evaluator (default: enabled).",
    )
    parser.add_argument("--generations", type=int, default=None, help="Override number of evolution generations.")
    parser.add_argument(
        "--enable-perturbation",
        dest="perturbation_enabled",
        action="store_true",
        default=None,
        help="Enable perturbation overrides for this run.",
    )
    parser.add_argument(
        "--disable-perturbation",
        dest="perturbation_enabled",
        action="store_false",
        help="Disable perturbation overrides for this run.",
    )
    parser.add_argument("--perturbation-trigger-generation", type=int, default=None)
    parser.add_argument("--persist-perturbation", action="store_true")
    add_two_window_fault_args(parser)
    args = parser.parse_args(argv)

    if two_window_fault_enabled(args) and (
        args.functional_ablation_mode != "full" or args.strategy_mode != "full"
    ):
        parser.error("The two-window robustness and component ablation protocols cannot be combined.")

    apply_model_overrides(config, args)
    config.egt.seed = args.seed
    config.egt.functional_ablation_mode = args.functional_ablation_mode
    configure_ablation_strategy_space(config.egt, args.strategy_mode)
    if args.enable_peer_context:
        config.egt.enable_peer_context = True
    config.egt.peer_context_max_chars = int(args.peer_context_max_chars)
    config.egt.risk_estimator_mode = str(args.risk_estimator_mode)
    if args.beta_mode is not None:
        config.egt.beta_mode = str(args.beta_mode)
    if args.fitness_penalty_beta is not None:
        config.egt.fitness_penalty_beta = float(args.fitness_penalty_beta)
    if args.adaptive_beta_quantile is not None:
        config.egt.adaptive_beta_quantile = float(args.adaptive_beta_quantile)
    if args.disable_cache:
        config.llm.disable_cache = True
    if args.generations is not None:
        config.egt.num_generations = int(args.generations)
    if args.perturbation_enabled is not None:
        config.perturbation.enabled = bool(args.perturbation_enabled)
    if args.perturbation_trigger_generation is not None:
        config.perturbation.trigger_generation = int(args.perturbation_trigger_generation)
    if args.persist_perturbation:
        config.perturbation.persist_across_gens = True
    if config.perturbation.enabled and not two_window_fault_enabled(args):
        trigger = int(config.perturbation.trigger_generation)
        if trigger < 0 or trigger >= config.egt.num_generations:
            parser.error(
                f"Perturbation generation {trigger} is outside the configured "
                f"0..{config.egt.num_generations - 1} range; increase --generations "
                "or change --perturbation-trigger-generation."
            )
    validate_two_window_fixed_horizon(
        enabled=two_window_fault_enabled(args),
        configured_rounds=config.egt.num_generations,
    )
    if two_window_fault_enabled(args) and config.perturbation.enabled:
        raise ValueError("Two-window and generation-triggered perturbations must run separately.")
    config.egt.prompt_style = args.prompt_style
    if args.disable_gene_role_prompts:
        config.egt.use_gene_role_prompts = False
    set_global_seed(args.seed)
    print("========== Starting Evolution Experiment ==========")
    print(f"[Seed] Using seed={args.seed}")
    print(f"[Run] Peer message context enabled={bool(config.egt.enable_peer_context)}")
    print(f"[Run] risk_estimator_mode={config.egt.risk_estimator_mode}")
    print(
        f"[Run] beta_mode={config.egt.beta_mode}, fitness_penalty_beta={float(config.egt.fitness_penalty_beta):.4f}, "
        f"adaptive_beta_quantile={float(config.egt.adaptive_beta_quantile):.2f}"
    )
    print(f"[Run] Perturbation enabled={bool(config.perturbation.enabled)}")
    print(f"[Run] prompt_style={config.egt.prompt_style}, gene_role_prompts={bool(config.egt.use_gene_role_prompts)}")
    print(f"[Run] Final-layer Judge enabled={bool(args.final_judge)}")

    jsonl_file = args.input_graph
    print(f"[Data] Loading graph from {jsonl_file}...")
    real_graph = load_real_graph(jsonl_file, line_idx=args.line_idx)
    validate_two_window_protocol_inputs(args=args, samples=[(args.line_idx, real_graph)])

    print("[Run] Starting co-evolution simulation...")
    calibrator_env = EvolutionEnvironment(real_graph, num_generations=config.egt.num_generations, config=config)
    calibration = apply_fixed_rule_parameter_calibration(config, calibrator_env)
    calibrator_env.close_debug_file()
    if calibration.get("applied"):
        print(
            "[Calib] Fixed-rule applied | "
            f"selection_alpha=({calibration['selection_alpha_start']:.4f}->{calibration['selection_alpha_end']:.4f}), "
            f"verify={calibration['verify_cost']:.6f}, "
            f"search={calibration['search_cost']:.6f}, "
            f"noise={calibration['exploration_noise']:.4f}, "
            f"mutation=({calibration['mutation_start']:.4f}->{calibration['mutation_end']:.4f}), "
            f"persona_rate={calibration['persona_mutation_rate']:.4f}, "
            f"edge_samples={calibration['edge_samples']}"
        )
    else:
        print(
            "[Calib] Fixed-rule skipped | "
            f"reason={calibration.get('calibration_reason', calibration.get('reason', 'unknown'))}, "
            f"selection_alpha=({calibration.get('selection_alpha_start', config.egt.selection_alpha_start):.4f}"
            f"->{calibration.get('selection_alpha_end', config.egt.selection_alpha_end):.4f}), "
            f"verify={float(calibration.get('verify_cost', config.egt.verify_cost)):.6f}"
        )

    fault_controller = build_two_window_fault_controller(
        args=args,
        sample_idx=args.line_idx,
        sample=real_graph,
    )
    run_result = run_single_episode(
        real_graph,
        config,
        calibration_summary=calibration,
        enable_runtime_recalibration=bool(getattr(config.egt, "enable_runtime_recalibration", False)),
        final_judge=bool(args.final_judge),
        fault_controller=fault_controller,
        return_env=True,
    )
    env = run_result.pop("env")
    history_stats = run_result["history_stats"]
    print("[Run] Generating figure and exporting results...")
    figure_output_path = args.output_figure or (args.output_json.with_suffix(".png") if args.output_json else None)
    plot_evolution_metrics(history_stats, agent_history=env.agent_history, config=config, output_path=figure_output_path)
    result_output_path = args.output_json or _path_with_model_folder(
        config.paths.experiment_results_path, config.llm.generation_model
    )
    extra_results = {
        key: value for key, value in run_result.items() if key.startswith("two_window_fault_")
    }
    extra_results.update({
        "sample_idx": args.line_idx,
        "question_id": real_graph.get("question_id") or real_graph.get("instance_id"),
        "initial_answers": real_graph.get("initial_answers"),
        "termination_reason": run_result["termination_reason"],
    })
    export_experiment_results(history_stats, env, output_path=result_output_path, extra_results=extra_results)
    env.close_debug_file()
    print("[Run] Experiment completed.")


if __name__ == "__main__":
    main()
