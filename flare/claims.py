from __future__ import annotations

from typing import Iterable, List, Sequence

from .core import Agent, ExecutionGraph, Node, NodeStatus, NodeType
from .detection import split_atomic_claims, supported_claim


def fclaim(answer: str) -> List[str]:
    return split_atomic_claims(answer)


def add_claim_nodes(
    graph: ExecutionGraph,
    answer: str,
    source_node: str | None = None,
    supported_by: Sequence[str] | None = None,
    start_time: float | None = None,
) -> List[str]:
    claim_ids: List[str] = []
    base_time = start_time if start_time is not None else _next_timestamp(graph)
    for index, claim in enumerate(fclaim(answer), start=1):
        claim_id = _unique_id(graph, f"claim_{index}")
        evidence = list(supported_by or [])
        graph.add_node(
            Node(
                id=claim_id,
                type=NodeType.CLAIM,
                content={"claim": claim},
                agent=Agent.SUMMARIZER,
                status=NodeStatus.SUCCESS,
                timestamp=base_time + index * 0.001,
                meta={"supported_by": evidence} if evidence else {},
            )
        )
        if source_node:
            graph.add_edge(source_node, claim_id, "answer_contains_claim", True)
        for evidence_id in evidence:
            graph.add_edge(evidence_id, claim_id, "observation_supports_claim", True)
        claim_ids.append(claim_id)
    return claim_ids


def claim_nodes(graph: ExecutionGraph) -> List[str]:
    return [node_id for node_id, node in graph.nodes.items() if node.type == NodeType.CLAIM]


def unsupported_claim_nodes(graph: ExecutionGraph) -> List[str]:
    return [node_id for node_id in claim_nodes(graph) if not supported_claim(graph, graph.nodes[node_id])]


def _next_timestamp(graph: ExecutionGraph) -> float:
    return max((node.timestamp for node in graph.nodes.values()), default=0.0) + 1.0


def _unique_id(graph: ExecutionGraph, prefix: str) -> str:
    if prefix not in graph.nodes:
        return prefix
    index = 2
    while f"{prefix}_{index}" in graph.nodes:
        index += 1
    return f"{prefix}_{index}"
