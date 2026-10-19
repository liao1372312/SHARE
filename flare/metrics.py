from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from .claims import unsupported_claim_nodes
from .pipeline import FLAREResult

EPSILON = 1e-9


@dataclass(frozen=True)
class LocalizationCase:
    gold_agent: str
    pred_agent: str
    gold_failure_type: str
    pred_failure_type: str
    gold_decision: str
    ranked_decisions: Sequence[str]


@dataclass(frozen=True)
class RepairCase:
    failed_before: bool
    success_after: bool
    unsupported_before: int = 0
    unsupported_after: int = 0
    affected_size: int = 0
    graph_size: int = 0
    extra_llm_calls: float = 0.0
    extra_api_calls: float = 0.0
    token_cost: float = 0.0
    latency: float = 0.0


def agent_attribution_accuracy(cases: Iterable[LocalizationCase]) -> float:
    items = list(cases)
    return _mean([case.gold_agent == case.pred_agent for case in items])


def failure_type_accuracy(cases: Iterable[LocalizationCase]) -> float:
    items = list(cases)
    return _mean([case.gold_failure_type == case.pred_failure_type for case in items])


def faulty_decision_accuracy(cases: Iterable[LocalizationCase]) -> float:
    items = list(cases)
    return _mean([bool(case.ranked_decisions) and case.ranked_decisions[0] == case.gold_decision for case in items])


def top2_recall(cases: Iterable[LocalizationCase]) -> float:
    items = list(cases)
    return _mean([case.gold_decision in list(case.ranked_decisions)[:2] for case in items])


def mrr(cases: Iterable[LocalizationCase]) -> float:
    values: List[float] = []
    for case in cases:
        ranked = list(case.ranked_decisions)
        values.append(1.0 / (ranked.index(case.gold_decision) + 1) if case.gold_decision in ranked else 0.0)
    return sum(values) / len(values) if values else 0.0


def repair_success_rate(cases: Iterable[RepairCase]) -> float:
    failed = [case for case in cases if case.failed_before]
    return _mean([case.success_after for case in failed])


def execution_success_rate_after_repair(cases: Iterable[RepairCase]) -> float:
    items = list(cases)
    return _mean([case.success_after for case in items])


def recovery_rate(cases: Iterable[RepairCase]) -> float:
    failed = [case for case in cases if case.failed_before]
    return _mean([case.success_after for case in failed])


def unsupported_claim_reduction(before: int, after: int, epsilon: float = EPSILON) -> float:
    return (before - after) / (before + epsilon)


def reexecution_ratio(affected_size: int, graph_size: int, epsilon: float = EPSILON) -> float:
    return affected_size / (graph_size + epsilon)


def aggregate_repair_cost(cases: Iterable[RepairCase]) -> Mapping[str, float]:
    items = list(cases)
    return {
        "reexecution_ratio": _avg([reexecution_ratio(case.affected_size, case.graph_size) for case in items]),
        "extra_llm_calls": _avg([case.extra_llm_calls for case in items]),
        "extra_api_calls": _avg([case.extra_api_calls for case in items]),
        "token_cost": _avg([case.token_cost for case in items]),
        "latency": _avg([case.latency for case in items]),
        "unsupported_claim_reduction": _avg([unsupported_claim_reduction(case.unsupported_before, case.unsupported_after) for case in items]),
    }


def repair_case_from_result(result: FLAREResult) -> RepairCase:
    affected = sum(len(iteration.localization.affected) for iteration in result.iterations if iteration.localization is not None)
    extra_llm = sum(1 for iteration in result.iterations if iteration.localization is not None)
    extra_api = sum(1 for iteration in result.iterations if iteration.action.value in {"RetryTool", "ReplaceTool", "AcquireEvidence"})
    return RepairCase(
        failed_before=bool(result.initial_failures),
        success_after=result.repaired,
        unsupported_before=result.unsupported_before,
        unsupported_after=result.unsupported_after,
        affected_size=affected,
        graph_size=len(result.graph.nodes),
        extra_llm_calls=float(extra_llm),
        extra_api_calls=float(extra_api),
    )


def _mean(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1.0 for item in items if item) / len(items) if items else 0.0


def _avg(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
