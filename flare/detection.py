from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .core import ExecutionGraph, FailureType, Node, NodeStatus, NodeType

_ERROR_KEYS = {"error", "exception", "timeout", "http_error", "service_unavailable", "empty_response", "invalid_return_type"}
_SCHEMA_KEYS = {"schema_error", "schema_mismatch", "invalid_schema", "type_error", "missing_required", "missing_required_parameters"}
_PARAM_KEYS = {"param_error", "parameter_error", "missing_parameter", "invalid_parameter", "missing_required", "missing_required_parameters"}
_DEP_KEYS = {"missing_dep", "missing_dependency", "invalid_dependency", "dependency_error", "unresolved_dependency"}
_API_KEYS = {"api_error", "wrong_api", "hallucinated_api", "unavailable_api", "api_unavailable"}
_UNRECOVERED_KEYS = {"unrecovered", "unresolved", "repair_failed", "repeated_retry"}


def detect_failures(graph: ExecutionGraph) -> Set[str]:
    failures: Set[str] = set()
    for node_id, node in graph.nodes.items():
        failure_types = failure_types_for_node(graph, node)
        if failure_types:
            node.meta["failure_types"] = sorted(item.value for item in failure_types)
            failures.add(node_id)
    return failures


def failure_types_for_node(graph: ExecutionGraph, node: Node) -> Set[FailureType]:
    meta = _flatten_meta(node.meta)
    text = _node_text(node).lower()
    failures: Set[FailureType] = set()
    if node.status == NodeStatus.FAILED or _has_any(meta, _ERROR_KEYS) or any(token in text for token in ("timeout", "http error", "service unavailable")):
        failures.add(FailureType.EXECUTION)
    if _has_any(meta, _SCHEMA_KEYS) or "schema mismatch" in text or "invalid return type" in text:
        failures.add(FailureType.SCHEMA)
    if _has_any(meta, _PARAM_KEYS) or "missing required" in text or "invalid parameter" in text:
        failures.add(FailureType.PARAMETER)
    if _has_any(meta, _DEP_KEYS) or _missing_dependencies(graph, node):
        failures.add(FailureType.MISSING_DEPENDENCY)
    if _has_any(meta, _API_KEYS) or "hallucinated api" in text or "wrong api" in text:
        failures.add(FailureType.API)
    if _has_any(meta, _UNRECOVERED_KEYS):
        failures.add(FailureType.UNRECOVERED)
    if node.type == NodeType.CLAIM and not supported_claim(graph, node):
        failures.add(FailureType.UNSUPPORTED_CLAIM)
    return failures


def supported_claim(graph: ExecutionGraph, claim: Node) -> bool:
    if claim.type != NodeType.CLAIM:
        return True
    if claim.meta.get("unsupported") is True:
        return False
    evidence = claim.meta.get("supported_by") or claim.meta.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if evidence:
        return any(eid in graph.nodes and graph.nodes[eid].status == NodeStatus.SUCCESS for eid in evidence)
    ancestors = graph.ancestors(claim.id)
    for node_id in ancestors:
        node = graph.nodes[node_id]
        if node.type == NodeType.OBSERVATION and node.status == NodeStatus.SUCCESS and graph.path_valid(node_id, claim.id):
            return True
    return bool(claim.meta.get("supported") is True)


def split_atomic_claims(answer: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+|[;；]\s*", answer.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _missing_dependencies(graph: ExecutionGraph, node: Node) -> bool:
    required = node.meta.get("requires") or node.meta.get("required_dependencies") or []
    if isinstance(required, str):
        required = [required]
    if required and not set(map(str, required)).issubset(graph.ancestors(node.id) | graph.predecessors(node.id)):
        return True
    if node.type in {NodeType.SERVICE_SELECTION, NodeType.PARAMETER_BINDING, NodeType.API_INVOCATION, NodeType.OBSERVATION}:
        return not graph.predecessors(node.id)
    return False


def _flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(meta or {})
    for key in ("validation", "constraints", "errors"):
        value = meta.get(key) if meta else None
        if isinstance(value, dict):
            flat.update(value)
    return flat


def _has_any(meta: Dict[str, Any], keys: Iterable[str]) -> bool:
    for key in keys:
        value = meta.get(key)
        if value is True:
            return True
        if isinstance(value, (str, list, tuple, dict)) and bool(value):
            return True
    return False


def _node_text(node: Node) -> str:
    return f"{node.content} {node.meta}"
