from flare import FLARE, ExecutionGraph
from flare.claims import add_claim_nodes, fclaim, unsupported_claim_nodes
from flare.backbone import MultiAgentComposer
from flare.core import Agent, Event, NodeStatus, NodeType
from flare.detection import detect_failures, supported_claim
from flare.localization import localize_faulty_decision
from flare.metrics import (
    LocalizationCase,
    agent_attribution_accuracy,
    faulty_decision_accuracy,
    mrr,
    reexecution_ratio,
    repair_case_from_result,
    repair_success_rate,
    top2_recall,
    unsupported_claim_reduction,
)
from flare.predicates import exec_err, fail, missing_dep, param_err, schema_err, unsupported_claim
from flare.repair import invalidated_nodes, merge_repaired_slice
from flare.slicing import merged_backward_slice


def failing_events():
    return [
        Event("q1", Agent.SYSTEM, NodeType.REQUEST, output="task", status=NodeStatus.SUCCESS, time=1),
        Event("p1", Agent.PLANNER, NodeType.PLAN, output="call weather", meta={"depends_on": ["q1"]}, time=2),
        Event("s1", Agent.SERVICE_PROVIDER, NodeType.SERVICE_SELECTION, output="WeatherAPI", meta={"depends_on": ["p1"]}, time=3),
        Event("b1", Agent.EXECUTOR, NodeType.PARAMETER_BINDING, output={"city": None}, meta={"depends_on": ["s1"], "missing_required": ["city"]}, status=NodeStatus.FAILED, time=4),
        Event("e1", Agent.EXECUTOR, NodeType.API_INVOCATION, output={"error": "missing city"}, meta={"depends_on": ["b1"], "error": "missing city"}, status=NodeStatus.FAILED, time=5),
        Event("c1", Agent.SUMMARIZER, NodeType.CLAIM, output="Weather is sunny", meta={"depends_on": ["e1"], "unsupported": True}, time=6),
    ]


def test_detect_slice_and_localize_parameter_root_cause():
    graph = ExecutionGraph.from_events(failing_events())
    failures = detect_failures(graph)
    assert {"b1", "e1", "c1"}.issubset(failures)
    slice_nodes = merged_backward_slice(graph, failures)
    assert {"q1", "p1", "s1", "b1", "e1", "c1"}.issubset(slice_nodes)
    loc = localize_faulty_decision(graph, failures, slice_nodes)
    assert loc is not None
    assert loc.node_id == "b1"
    assert loc.failure_type in {"parameter_error", "schema_error", "execution_error"}
    assert "e1" in loc.affected and "c1" in loc.affected


def test_bounded_repair_loop_returns_result():
    result = FLARE().run(failing_events())
    assert result.initial_failures
    assert result.iterations
    assert result.iterations[0].localization.node_id == "b1"
    assert result.iterations[0].action.value == "RebindParam"


def test_supported_claim_with_valid_observation_path():
    events = [
        Event("q", Agent.SYSTEM, NodeType.REQUEST, output="task", time=1),
        Event("obs", Agent.EXECUTOR, NodeType.OBSERVATION, output={"temperature": 20}, meta={"depends_on": ["q"]}, time=2),
        Event("claim", Agent.SUMMARIZER, NodeType.CLAIM, output="Temperature is 20", meta={"depends_on": ["obs"], "supported_by": ["obs"]}, time=3),
    ]
    graph = ExecutionGraph.from_events(events)
    assert supported_claim(graph, graph.nodes["claim"])
    assert detect_failures(graph) == set()


def test_graph_exposes_paper_g_t_a_s_and_node_sets():
    graph = ExecutionGraph.from_events(failing_events())
    assert graph.temporal_order == ["q1", "p1", "s1", "b1", "e1", "c1"]
    assert graph.agent_map["p1"] == Agent.PLANNER
    assert graph.status_map["b1"] == NodeStatus.FAILED
    assert graph.node_sets[NodeType.REQUEST] == {"q1"}
    assert graph.node_sets[NodeType.PARAMETER_BINDING] == {"b1"}
    assert graph.depends("b1", "s1")
    assert graph.has_path("s1", "c1")


def test_slice_and_invalid_follow_path_definitions():
    graph = ExecutionGraph.from_events(failing_events())
    failures = detect_failures(graph)
    slice_nodes = merged_backward_slice(graph, failures)
    assert slice_nodes == {node_id for node_id in graph.nodes if any(graph.has_path(node_id, failure) for failure in failures)}
    invalid = invalidated_nodes(graph, "b1")
    assert invalid == {node_id for node_id in graph.nodes if node_id != "b1" and graph.has_path("b1", node_id)}


def test_merge_repaired_slice_matches_graph_merge():
    graph = ExecutionGraph.from_events(failing_events())
    base = graph.without_nodes({"e1", "c1"})
    repaired = ExecutionGraph()
    merged = merge_repaired_slice(base, repaired)
    assert set(merged.nodes) == set(base.nodes)


def test_paper_failure_predicates_are_explicit():
    graph = ExecutionGraph.from_events(failing_events())
    assert fail(graph, graph.nodes["b1"])
    assert param_err(graph, graph.nodes["b1"])
    assert exec_err(graph, graph.nodes["e1"])
    assert unsupported_claim(graph, graph.nodes["c1"])
    assert not schema_err(graph, graph.nodes["p1"])
    assert not missing_dep(graph, graph.nodes["p1"])


def test_fclaim_and_claim_node_generation():
    graph = ExecutionGraph.from_events([
        Event("obs", Agent.EXECUTOR, NodeType.OBSERVATION, output={"ok": True}, time=1),
    ])
    assert fclaim("A is true. B is true!") == ["A is true.", "B is true!"]
    claim_ids = add_claim_nodes(graph, "A is true. B is true!", supported_by=["obs"])
    assert len(claim_ids) == 2
    assert unsupported_claim_nodes(graph) == []


def test_paper_evaluation_metrics():
    cases = [
        LocalizationCase("executor", "executor", "parameter_error", "parameter_error", "b1", ["b1", "s1"]),
        LocalizationCase("planner", "executor", "missing_dependency", "execution_error", "p1", ["s1", "p1"]),
    ]
    assert agent_attribution_accuracy(cases) == 0.5
    assert faulty_decision_accuracy(cases) == 0.5
    assert top2_recall(cases) == 1.0
    assert mrr(cases) == 0.75
    assert unsupported_claim_reduction(4, 1) > 0.74
    assert reexecution_ratio(2, 8) == 2 / (8 + 1e-9)
    result = FLARE().run(failing_events())
    repair_case = repair_case_from_result(result)
    assert repair_success_rate([repair_case]) == 1.0
    assert repair_case.unsupported_before == 1


def test_end_to_end_backbone_repairs_missing_parameter():
    output = MultiAgentComposer().run("Get current weather for city Beijing", inject_failure="missing_param")
    assert output.result.repaired
    assert output.result.final_failures == set()
    assert "weather in Beijing" in output.final_answer
    assert output.result.iterations[0].action.value == "RebindParam"


def test_end_to_end_backbone_success_without_repair():
    output = MultiAgentComposer().run("Get current weather for city Beijing")
    assert output.result.repaired
    assert output.result.initial_failures == set()
    assert output.result.iterations == []
    assert "sunny" in output.final_answer
