from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from .core import Event, ExecutionGraph, RepairAction
from .claims import unsupported_claim_nodes
from .detection import detect_failures
from .localization import LocalizationResult, LocalizationWeights, localize_faulty_decision
from .repair import RepairHandler, RepairUtilityWeights, apply_minimal_repair, repair_succeeded, select_repair_action
from .slicing import merged_backward_slice


@dataclass
class FLAREConfig:
    repair_budget: int = 3
    localization_weights: LocalizationWeights = field(default_factory=LocalizationWeights)
    repair_weights: RepairUtilityWeights = field(default_factory=RepairUtilityWeights)


@dataclass
class RepairIteration:
    failures: Set[str]
    slice_nodes: Set[str]
    localization: Optional[LocalizationResult]
    action: RepairAction


@dataclass
class FLAREResult:
    graph: ExecutionGraph
    initial_failures: Set[str]
    final_failures: Set[str]
    iterations: List[RepairIteration]
    repaired: bool
    unsupported_before: int = 0
    unsupported_after: int = 0


class FLARE:
    def __init__(self, config: FLAREConfig | None = None, repair_handler: RepairHandler | None = None):
        self.config = config or FLAREConfig()
        self.repair_handler = repair_handler

    def build_graph(self, events: Iterable[Event | Dict[str, Any]]) -> ExecutionGraph:
        return ExecutionGraph.from_events(events)

    def run(self, events_or_graph: Iterable[Event | Dict[str, Any]] | ExecutionGraph) -> FLAREResult:
        graph = events_or_graph if isinstance(events_or_graph, ExecutionGraph) else self.build_graph(events_or_graph)
        initial_failures = detect_failures(graph)
        unsupported_before = len(unsupported_claim_nodes(graph))
        failures = set(initial_failures)
        iterations: List[RepairIteration] = []
        budget = self.config.repair_budget
        while failures and len(iterations) < budget:
            slice_nodes = merged_backward_slice(graph, failures)
            localization = localize_faulty_decision(graph, failures, slice_nodes, self.config.localization_weights)
            if localization is None:
                iterations.append(RepairIteration(failures, slice_nodes, None, RepairAction.NOOP))
                break
            action = select_repair_action(graph, localization, self.config.repair_weights)
            iterations.append(RepairIteration(set(failures), set(slice_nodes), localization, action))
            if action == RepairAction.NOOP:
                break
            graph = apply_minimal_repair(graph, localization, action, self.repair_handler)
            failures = detect_failures(graph)
        final_failures = detect_failures(graph)
        unsupported_after = len(unsupported_claim_nodes(graph))
        return FLAREResult(
            graph=graph,
            initial_failures=initial_failures,
            final_failures=final_failures,
            iterations=iterations,
            repaired=repair_succeeded(graph),
            unsupported_before=unsupported_before,
            unsupported_after=unsupported_after,
        )
