"""Check the known plural missing-parameters detector regression.

This is a local validation command for the human workbench task. A passing
result checks the specific detector behavior; it is not a workflow repair
success label for the paper.
"""
from __future__ import annotations

from flare.core import Agent, Event, ExecutionGraph, FailureType, NodeStatus, NodeType
from flare.detection import detect_failures


def main() -> None:
    graph = ExecutionGraph.from_events([
        Event("request", Agent.SYSTEM, NodeType.REQUEST, output="Bind the city", time=1),
        Event("binding", Agent.EXECUTOR, NodeType.PARAMETER_BINDING,
              output={"city": None}, meta={"depends_on": ["request"], "missing_parameters": ["city"]},
              status=NodeStatus.FAILED, time=2),
    ])
    detect_failures(graph)
    assert FailureType.PARAMETER in graph.nodes["binding"].failure_types, (
        "plural missing_parameters was not recognized as a parameter failure"
    )
    print("PASS: plural missing_parameters is detected as parameter_error")


if __name__ == "__main__":
    main()
