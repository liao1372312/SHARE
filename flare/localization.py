from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .core import ExecutionGraph, FailureType, Node, NodeType
from .detection import failure_types_for_node
from .slicing import affected_downstream_slice

DECISION_TYPES = {
    NodeType.PLAN,
    NodeType.SERVICE_SELECTION,
    NodeType.PARAMETER_BINDING,
    NodeType.API_INVOCATION,
    NodeType.REPAIR,
    NodeType.CLAIM,
}


@dataclass
class LocalizationWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    lambda_: float = 1.0
    mu: float = 0.2
    epsilon: float = 1e-9
    api_error: float = 1.0
    schema_error: float = 1.0
    param_error: float = 1.0
    evidence_error: float = 1.0
    dep_error: float = 1.0
    affected_cost: float = 0.05
    llm_call_cost: float = 0.2
    api_call_cost: float = 0.2
    latency_cost: float = 0.05


@dataclass
class LocalizationResult:
    agent: str
    node_id: str
    node_type: str
    failure_type: str
    confidence: float
    affected: Set[str]
    scores: Dict[str, float]


def candidate_faulty_decisions(graph: ExecutionGraph, slice_nodes: Iterable[str]) -> Set[str]:
    return {node_id for node_id in slice_nodes if graph.nodes[node_id].type in DECISION_TYPES}


def localize_faulty_decision(
    graph: ExecutionGraph,
    failure_nodes: Set[str],
    slice_nodes: Set[str],
    weights: LocalizationWeights | None = None,
) -> LocalizationResult | None:
    weights = weights or LocalizationWeights()
    candidates = candidate_faulty_decisions(graph, slice_nodes)
    if not candidates:
        candidates = set(slice_nodes)
    scores = {node_id: score_candidate(graph, node_id, failure_nodes, weights) for node_id in candidates}
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: (item[1], -graph.nodes[item[0]].timestamp), reverse=True)
    best_id, best_score = ranked[0]
    total_positive = sum(max(score, 0.0) for score in scores.values())
    confidence = max(best_score, 0.0) / total_positive if total_positive > 0 else 0.0
    node = graph.nodes[best_id]
    failures = node.failure_types or failure_types_for_node(graph, node)
    if not failures:
        downstream_failure_types = []
        for failure in failure_nodes:
            downstream_failure_types.extend(graph.nodes[failure].failure_types or failure_types_for_node(graph, graph.nodes[failure]))
        failure_type = _preferred_failure_type(graph.nodes[best_id], downstream_failure_types)
    else:
        failure_type = _preferred_failure_type(graph.nodes[best_id], failures)
    return LocalizationResult(
        agent=node.agent.value,
        node_id=best_id,
        node_type=node.type.value,
        failure_type=failure_type,
        confidence=confidence,
        affected=affected_downstream_slice(graph, best_id),
        scores=dict(ranked),
    )


def _preferred_failure_type(node: Node, failures: Iterable[FailureType]) -> str:
    values = set(failures)
    by_node_type = {
        NodeType.PLAN: [FailureType.MISSING_DEPENDENCY, FailureType.DEPENDENCY, FailureType.UNRECOVERED],
        NodeType.SERVICE_SELECTION: [FailureType.API, FailureType.SCHEMA],
        NodeType.PARAMETER_BINDING: [FailureType.PARAMETER, FailureType.SCHEMA, FailureType.MISSING_DEPENDENCY],
        NodeType.API_INVOCATION: [FailureType.EXECUTION],
        NodeType.REPAIR: [FailureType.UNRECOVERED],
        NodeType.CLAIM: [FailureType.UNSUPPORTED_CLAIM],
    }
    global_order = [
        *by_node_type.get(node.type, []),
        FailureType.PARAMETER,
        FailureType.API,
        FailureType.SCHEMA,
        FailureType.MISSING_DEPENDENCY,
        FailureType.UNSUPPORTED_CLAIM,
        FailureType.UNRECOVERED,
        FailureType.EXECUTION,
        FailureType.DEPENDENCY,
    ]
    for failure in global_order:
        if failure in values:
            return failure.value
    return sorted(item.value for item in values)[0] if values else "unknown"


def score_candidate(graph: ExecutionGraph, node_id: str, failure_nodes: Set[str], weights: LocalizationWeights) -> float:
    node = graph.nodes[node_id]
    upstream_failures = graph.ancestors(node_id) & failure_nodes
    propagated_claim_penalty = 1.5 if node.type == NodeType.CLAIM and upstream_failures else 0.0
    return (
        weights.alpha * error_score(graph, node_id)
        + weights.beta * impact_score(graph, node_id, failure_nodes, weights.epsilon)
        + weights.gamma * proximity_score(graph, node_id, failure_nodes)
        + weights.lambda_ * violation_score(graph, node_id, weights)
        - weights.mu * repair_cost(graph, node_id, weights)
        - propagated_claim_penalty
    )


def error_score(graph: ExecutionGraph, node_id: str) -> float:
    node = graph.nodes[node_id]
    failures = failure_types_for_node(graph, node)
    if failures:
        return 1.0
    meta = node.meta
    if meta.get("warning") or meta.get("low_confidence") or meta.get("retry_count", 0):
        return 0.5
    return 0.0


def impact_score(graph: ExecutionGraph, node_id: str, failure_nodes: Set[str], epsilon: float) -> float:
    descendants = graph.descendants(node_id)
    return len(descendants & failure_nodes) / (len(failure_nodes) + epsilon)


def proximity_score(graph: ExecutionGraph, node_id: str, failure_nodes: Set[str]) -> float:
    values: List[float] = []
    for failure in failure_nodes:
        distance = graph.shortest_distance(node_id, failure)
        if distance is not None:
            values.append(math.exp(-distance))
    return max(values) if values else 0.0


def violation_score(graph: ExecutionGraph, node_id: str, weights: LocalizationWeights) -> float:
    failures = failure_types_for_node(graph, graph.nodes[node_id])
    return (
        weights.api_error * (FailureType.API in failures)
        + weights.schema_error * (FailureType.SCHEMA in failures)
        + weights.param_error * (FailureType.PARAMETER in failures)
        + weights.evidence_error * (FailureType.UNSUPPORTED_CLAIM in failures)
        + weights.dep_error * (FailureType.MISSING_DEPENDENCY in failures or FailureType.DEPENDENCY in failures)
    )


def repair_cost(graph: ExecutionGraph, node_id: str, weights: LocalizationWeights) -> float:
    affected = affected_downstream_slice(graph, node_id)
    node = graph.nodes[node_id]
    llm_calls = float(node.meta.get("llm_calls", 1 if node.type in {NodeType.PLAN, NodeType.SERVICE_SELECTION, NodeType.PARAMETER_BINDING, NodeType.REPAIR, NodeType.CLAIM} else 0))
    api_calls = float(node.meta.get("api_calls", 1 if node.type == NodeType.API_INVOCATION else 0))
    latency = float(node.meta.get("latency", 0))
    return weights.affected_cost * len(affected) + weights.llm_call_cost * llm_calls + weights.api_call_cost * api_calls + weights.latency_cost * latency
