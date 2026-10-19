from __future__ import annotations

from .core import ExecutionGraph, FailureType, Node, NodeStatus, NodeType
from .detection import failure_types_for_node, supported_claim


def exec_err(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.EXECUTION in failure_types_for_node(graph, node)


def schema_err(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.SCHEMA in failure_types_for_node(graph, node)


def missing_dep(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.MISSING_DEPENDENCY in failure_types_for_node(graph, node)


def unsupported_claim(graph: ExecutionGraph, node: Node) -> bool:
    return node.type == NodeType.CLAIM and not supported_claim(graph, node)


def unrecovered(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.UNRECOVERED in failure_types_for_node(graph, node)


def fail(graph: ExecutionGraph, node: Node) -> bool:
    return exec_err(graph, node) or schema_err(graph, node) or missing_dep(graph, node) or unsupported_claim(graph, node) or unrecovered(graph, node)


def api_err(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.API in failure_types_for_node(graph, node)


def param_err(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.PARAMETER in failure_types_for_node(graph, node)


def evidence_err(graph: ExecutionGraph, node: Node) -> bool:
    return unsupported_claim(graph, node)


def dep_err(graph: ExecutionGraph, node: Node) -> bool:
    return FailureType.DEPENDENCY in failure_types_for_node(graph, node) or missing_dep(graph, node)


def repair_succ(graph: ExecutionGraph) -> bool:
    from .repair import repair_succeeded

    return repair_succeeded(graph)
