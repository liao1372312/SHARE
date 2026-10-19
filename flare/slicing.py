from __future__ import annotations

from typing import Set

from .core import ExecutionGraph


def backward_slice(graph: ExecutionGraph, failure_node: str) -> Set[str]:
    return {node_id for node_id in graph.nodes if graph.has_path(node_id, failure_node)}


def merged_backward_slice(graph: ExecutionGraph, failure_nodes: set[str]) -> Set[str]:
    merged: Set[str] = set()
    for failure in failure_nodes:
        merged |= backward_slice(graph, failure)
    return merged


def affected_downstream_slice(graph: ExecutionGraph, faulty_node: str) -> Set[str]:
    return {node_id for node_id in graph.nodes if node_id != faulty_node and graph.has_path(faulty_node, node_id)}
