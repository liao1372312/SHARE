from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set

from .core import ExecutionGraph, FailureType, Node, NodeStatus, NodeType, RepairAction
from .detection import detect_failures, supported_claim
from .localization import LocalizationResult
from .slicing import affected_downstream_slice

RepairHandler = Callable[[ExecutionGraph, str, RepairAction, Set[str]], ExecutionGraph]


@dataclass
class RepairUtilityWeights:
    success_prob: float = 1.0
    constraint_fix: float = 1.0
    evidence_gain: float = 1.0
    exec_cost: float = 0.2
    risk: float = 0.5


DEFAULT_ACTIONS = {
    FailureType.MISSING_DEPENDENCY: [RepairAction.ADD_SUBTASK, RepairAction.REPLAN],
    FailureType.DEPENDENCY: [RepairAction.ADD_SUBTASK, RepairAction.REPLAN],
    FailureType.API: [RepairAction.RESELECT_API, RepairAction.REPLACE_TOOL],
    FailureType.SCHEMA: [RepairAction.REBIND_PARAM, RepairAction.RESELECT_API],
    FailureType.PARAMETER: [RepairAction.REBIND_PARAM],
    FailureType.EXECUTION: [RepairAction.RETRY_TOOL, RepairAction.REPLACE_TOOL],
    FailureType.UNRECOVERED: [RepairAction.REVISE_REPAIR, RepairAction.REPLAN],
    FailureType.UNSUPPORTED_CLAIM: [RepairAction.REVISE_ANSWER, RepairAction.MARK_UNCERTAIN, RepairAction.ACQUIRE_EVIDENCE],
}


def select_repair_action(
    graph: ExecutionGraph,
    localization: LocalizationResult,
    weights: RepairUtilityWeights | None = None,
) -> RepairAction:
    weights = weights or RepairUtilityWeights()
    failure_type = _coerce_failure_type(localization.failure_type)
    actions = DEFAULT_ACTIONS.get(failure_type, [RepairAction.NOOP])
    utilities = {action: repair_utility(graph, localization.node_id, action, failure_type, weights) for action in actions}
    return max(utilities.items(), key=lambda item: item[1])[0]


def repair_utility(
    graph: ExecutionGraph,
    node_id: str,
    action: RepairAction,
    failure_type: FailureType | None,
    weights: RepairUtilityWeights,
) -> float:
    success = _success_probability(action, failure_type)
    constraint = _constraint_fix(action, failure_type)
    evidence = 1.0 if action in {RepairAction.ACQUIRE_EVIDENCE, RepairAction.REVISE_ANSWER, RepairAction.MARK_UNCERTAIN} else 0.0
    cost = _exec_cost(graph, node_id, action)
    risk = _risk(graph, node_id, action)
    return weights.success_prob * success + weights.constraint_fix * constraint + weights.evidence_gain * evidence - weights.exec_cost * cost - weights.risk * risk


def apply_minimal_repair(
    graph: ExecutionGraph,
    localization: LocalizationResult,
    action: RepairAction,
    handler: RepairHandler | None = None,
) -> ExecutionGraph:
    invalid = invalidated_nodes(graph, localization.node_id)
    base = graph.without_nodes(invalid)
    repaired_slice = execute_repair(action, localization.node_id, base, invalid, handler)
    return merge_repaired_slice(base, repaired_slice)


def invalidated_nodes(graph: ExecutionGraph, faulty_node: str) -> Set[str]:
    return affected_downstream_slice(graph, faulty_node)


def execute_repair(
    action: RepairAction,
    faulty_node: str,
    context_graph: ExecutionGraph,
    invalid: Set[str],
    handler: RepairHandler | None = None,
) -> ExecutionGraph:
    if handler:
        return handler(context_graph, faulty_node, action, invalid)
    return _symbolic_repair(context_graph, faulty_node, action, invalid)


def merge_repaired_slice(base: ExecutionGraph, repaired_slice: ExecutionGraph) -> ExecutionGraph:
    return base.merge(repaired_slice)


def repair_succeeded(graph: ExecutionGraph) -> bool:
    failures = detect_failures(graph)
    if failures:
        return False
    for node in graph.nodes.values():
        if node.type == NodeType.CLAIM and not supported_claim(graph, node):
            return False
    return True


def _symbolic_repair(base: ExecutionGraph, node_id: str, action: RepairAction, affected: Set[str]) -> ExecutionGraph:
    graph = ExecutionGraph(dict(base.nodes), set(base.edges))
    if node_id not in graph.nodes:
        return graph
    node = graph.nodes[node_id]
    node.status = NodeStatus.REPAIRED
    node.meta["repair_action"] = action.value
    node.meta.pop("failure_types", None)
    for key in [
        "error",
        "exception",
        "timeout",
        "schema_error",
        "schema_mismatch",
        "param_error",
        "parameter_error",
        "missing_parameter",
        "missing_required",
        "missing_dependency",
        "wrong_api",
        "hallucinated_api",
        "unrecovered",
        "unsupported",
    ]:
        node.meta.pop(key, None)
    if action == RepairAction.MARK_UNCERTAIN:
        node.status = NodeStatus.UNCERTAIN
        node.meta["supported"] = True
        node.meta["uncertain"] = True
    if action == RepairAction.REVISE_ANSWER and node.type == NodeType.CLAIM:
        node.meta["supported"] = True
        node.meta["revised"] = True
    if action == RepairAction.ACQUIRE_EVIDENCE and node.type == NodeType.CLAIM:
        obs_id = f"repair_obs_for_{node.id}"
        graph.add_node(
            Node(
                id=obs_id,
                type=NodeType.OBSERVATION,
                content={"repair_evidence_for": node.id},
                agent=node.agent,
                status=NodeStatus.SUCCESS,
                timestamp=node.timestamp + 0.001,
                meta={"generated_by_repair": True},
            )
        )
        graph.add_edge(obs_id, node.id, "repair_evidence", True)
        node.meta["supported_by"] = [obs_id]
    return graph


def _coerce_failure_type(value: str) -> Optional[FailureType]:
    try:
        return FailureType(value)
    except ValueError:
        return None


def _success_probability(action: RepairAction, failure_type: FailureType | None) -> float:
    if failure_type is None:
        return 0.0
    preferred = DEFAULT_ACTIONS.get(failure_type, [])
    if not preferred:
        return 0.2
    if action == preferred[0]:
        return 1.0
    if action in preferred:
        return 0.75
    return 0.25


def _constraint_fix(action: RepairAction, failure_type: FailureType | None) -> float:
    if failure_type in {FailureType.PARAMETER, FailureType.SCHEMA} and action == RepairAction.REBIND_PARAM:
        return 1.0
    if failure_type == FailureType.API and action in {RepairAction.RESELECT_API, RepairAction.REPLACE_TOOL}:
        return 1.0
    if failure_type in {FailureType.MISSING_DEPENDENCY, FailureType.DEPENDENCY} and action in {RepairAction.ADD_SUBTASK, RepairAction.REPLAN}:
        return 1.0
    if failure_type == FailureType.UNSUPPORTED_CLAIM and action in {RepairAction.REVISE_ANSWER, RepairAction.MARK_UNCERTAIN, RepairAction.ACQUIRE_EVIDENCE}:
        return 1.0
    if failure_type == FailureType.EXECUTION and action in {RepairAction.RETRY_TOOL, RepairAction.REPLACE_TOOL}:
        return 1.0
    if failure_type == FailureType.UNRECOVERED and action == RepairAction.REVISE_REPAIR:
        return 1.0
    return 0.0


def _exec_cost(graph: ExecutionGraph, node_id: str, action: RepairAction) -> float:
    affected = len(affected_downstream_slice(graph, node_id))
    action_cost = {
        RepairAction.REVISE_ANSWER: 0.5,
        RepairAction.MARK_UNCERTAIN: 0.2,
        RepairAction.REBIND_PARAM: 1.0,
        RepairAction.RETRY_TOOL: 1.0,
        RepairAction.RESELECT_API: 1.5,
        RepairAction.ADD_SUBTASK: 2.0,
        RepairAction.REPLAN: 3.0,
        RepairAction.REPLACE_TOOL: 2.0,
        RepairAction.ACQUIRE_EVIDENCE: 2.0,
        RepairAction.REVISE_REPAIR: 1.5,
        RepairAction.NOOP: 0.0,
    }[action]
    return action_cost + affected * 0.1


def _risk(graph: ExecutionGraph, node_id: str, action: RepairAction) -> float:
    node = graph.nodes[node_id]
    previous = node.meta.get("repair_action")
    repeat_penalty = 1.0 if previous == action.value else 0.0
    broad_penalty = 0.5 if action == RepairAction.REPLAN else 0.0
    return repeat_penalty + broad_penalty
