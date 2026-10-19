from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class Agent(str, Enum):
    PLANNER = "planner"
    SERVICE_PROVIDER = "service_provider"
    EXECUTOR = "executor"
    SUPERVISOR = "supervisor"
    SUMMARIZER = "summarizer"
    SYSTEM = "system"


class NodeType(str, Enum):
    REQUEST = "request"
    PLAN = "plan"
    SERVICE_SELECTION = "service_selection"
    PARAMETER_BINDING = "parameter_binding"
    API_INVOCATION = "api_invocation"
    OBSERVATION = "observation"
    REPAIR = "repair"
    CLAIM = "claim"


class NodeStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    INVALIDATED = "invalidated"
    REPAIRED = "repaired"
    UNCERTAIN = "uncertain"


class FailureType(str, Enum):
    EXECUTION = "execution_error"
    SCHEMA = "schema_error"
    MISSING_DEPENDENCY = "missing_dependency"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    UNRECOVERED = "unrecovered"
    API = "api_error"
    PARAMETER = "parameter_error"
    DEPENDENCY = "dependency_error"


class RepairAction(str, Enum):
    REPLAN = "Replan"
    ADD_SUBTASK = "AddSubtask"
    RESELECT_API = "ReselectAPI"
    REBIND_PARAM = "RebindParam"
    RETRY_TOOL = "RetryTool"
    REPLACE_TOOL = "ReplaceTool"
    REVISE_REPAIR = "ReviseRepair"
    REVISE_ANSWER = "ReviseAnswer"
    MARK_UNCERTAIN = "MarkUncertain"
    ACQUIRE_EVIDENCE = "AcquireEvidence"
    NOOP = "NoOp"


@dataclass
class Event:
    id: str
    agent: Agent | str
    type: NodeType | str
    input: Any = None
    output: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    status: NodeStatus | str = NodeStatus.SUCCESS
    time: float = 0.0


@dataclass
class Node:
    id: str
    type: NodeType
    content: Any
    agent: Agent
    status: NodeStatus
    timestamp: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def failure_types(self) -> Set[FailureType]:
        values = self.meta.get("failure_types", [])
        return {FailureType(value) if not isinstance(value, FailureType) else value for value in values}


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str = "depends"
    valid: bool = True


@dataclass
class ExecutionGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Set[Edge] = field(default_factory=set)

    @property
    def temporal_order(self) -> List[str]:
        return [node.id for node in sorted(self.nodes.values(), key=lambda item: (item.timestamp, item.id))]

    @property
    def agent_map(self) -> Dict[str, Agent]:
        return {node_id: node.agent for node_id, node in self.nodes.items()}

    @property
    def status_map(self) -> Dict[str, NodeStatus]:
        return {node_id: node.status for node_id, node in self.nodes.items()}

    @property
    def node_sets(self) -> Dict[NodeType, Set[str]]:
        groups = {node_type: set() for node_type in NodeType}
        for node_id, node in self.nodes.items():
            groups[node.type].add(node_id)
        return groups

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, source: str, target: str, relation: str = "depends", valid: bool = True) -> None:
        if source in self.nodes and target in self.nodes and source != target:
            self.edges.add(Edge(source, target, relation, valid))

    def depends(self, target: str, source: str, valid_only: bool = False) -> bool:
        return any(edge.source == source and edge.target == target and (edge.valid or not valid_only) for edge in self.edges)

    def has_path(self, source: str, target: str, valid_only: bool = False) -> bool:
        if source == target:
            return True
        stack = list(self.successors(source, valid_only=valid_only))
        seen = set(stack)
        while stack:
            current = stack.pop()
            if current == target:
                return True
            for nxt in self.successors(current, valid_only=valid_only):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def path_valid(self, source: str, target: str) -> bool:
        return self.has_path(source, target, valid_only=True)

    def predecessors(self, node_id: str, valid_only: bool = False) -> Set[str]:
        return {e.source for e in self.edges if e.target == node_id and (e.valid or not valid_only)}

    def successors(self, node_id: str, valid_only: bool = False) -> Set[str]:
        return {e.target for e in self.edges if e.source == node_id and (e.valid or not valid_only)}

    def ancestors(self, node_id: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self.predecessors(node_id))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.predecessors(current) - seen)
        return seen

    def descendants(self, node_id: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self.successors(node_id))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.successors(current) - seen)
        return seen

    def shortest_distance(self, source: str, target: str) -> Optional[int]:
        if source == target:
            return 0
        queue: List[Tuple[str, int]] = [(source, 0)]
        seen = {source}
        for current, distance in queue:
            for nxt in self.successors(current):
                if nxt == target:
                    return distance + 1
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        return None

    def induced_subgraph(self, node_ids: Iterable[str]) -> "ExecutionGraph":
        selected = set(node_ids)
        graph = ExecutionGraph()
        for node_id in selected:
            if node_id in self.nodes:
                graph.add_node(self.nodes[node_id])
        for edge in self.edges:
            if edge.source in selected and edge.target in selected:
                graph.edges.add(edge)
        return graph

    def without_nodes(self, node_ids: Iterable[str]) -> "ExecutionGraph":
        removed = set(node_ids)
        graph = ExecutionGraph()
        for node_id, node in self.nodes.items():
            if node_id not in removed:
                graph.add_node(node)
        for edge in self.edges:
            if edge.source not in removed and edge.target not in removed:
                graph.edges.add(edge)
        return graph

    def merge(self, other: "ExecutionGraph") -> "ExecutionGraph":
        graph = ExecutionGraph(dict(self.nodes), set(self.edges))
        graph.nodes.update(other.nodes)
        graph.edges |= other.edges
        return graph

    @classmethod
    def from_events(cls, events: Iterable[Event | Dict[str, Any]]) -> "ExecutionGraph":
        graph = cls()
        normalized = [_coerce_event(event) for event in events]
        for event in sorted(normalized, key=lambda item: (item.time, item.id)):
            graph.add_node(
                Node(
                    id=event.id,
                    type=_coerce_node_type(event.type),
                    content={"input": event.input, "output": event.output},
                    agent=_coerce_agent(event.agent),
                    status=_coerce_status(event.status),
                    timestamp=event.time,
                    meta=dict(event.meta or {}),
                )
            )
        explicit_edges = False
        for event in normalized:
            meta = event.meta or {}
            for parent in meta.get("depends_on", []) + meta.get("parents", []):
                graph.add_edge(str(parent), event.id, meta.get("relation", "depends"), True)
                explicit_edges = True
            for edge in meta.get("edges", []):
                graph.add_edge(str(edge[0]), str(edge[1]), edge[2] if len(edge) > 2 else "depends", True)
                explicit_edges = True
        if not explicit_edges:
            _add_rule_based_edges(graph)
        return graph


def _coerce_event(event: Event | Dict[str, Any]) -> Event:
    if isinstance(event, Event):
        return event
    return Event(
        id=str(event.get("id") or event.get("idt") or event.get("node_id")),
        agent=event.get("agent") or event.get("agentt") or Agent.SYSTEM,
        type=event.get("type") or event.get("typet"),
        input=event.get("input") if "input" in event else event.get("inputt"),
        output=event.get("output") if "output" in event else event.get("outputt"),
        meta=dict(event.get("meta") or event.get("metat") or {}),
        status=event.get("status") or event.get("statust") or NodeStatus.SUCCESS,
        time=float(event.get("time") or event.get("timet") or event.get("timestamp") or 0.0),
    )


def _coerce_node_type(value: NodeType | str) -> NodeType:
    if isinstance(value, NodeType):
        return value
    text = str(value).lower()
    aliases = {
        "planning": NodeType.PLAN,
        "subtask": NodeType.PLAN,
        "selection": NodeType.SERVICE_SELECTION,
        "api_selection": NodeType.SERVICE_SELECTION,
        "binding": NodeType.PARAMETER_BINDING,
        "parameters": NodeType.PARAMETER_BINDING,
        "invoke": NodeType.API_INVOCATION,
        "tool_call": NodeType.API_INVOCATION,
        "api_call": NodeType.API_INVOCATION,
        "answer_claim": NodeType.CLAIM,
        "final_claim": NodeType.CLAIM,
    }
    return aliases.get(text, NodeType(text))


def _coerce_agent(value: Agent | str) -> Agent:
    if isinstance(value, Agent):
        return value
    text = str(value).lower()
    aliases = {"provider": Agent.SERVICE_PROVIDER, "service": Agent.SERVICE_PROVIDER, "exec": Agent.EXECUTOR, "summary": Agent.SUMMARIZER}
    return aliases.get(text, Agent(text) if text in {item.value for item in Agent} else Agent.SYSTEM)


def _coerce_status(value: NodeStatus | str) -> NodeStatus:
    if isinstance(value, NodeStatus):
        return value
    text = str(value).lower()
    aliases = {"ok": NodeStatus.SUCCESS, "error": NodeStatus.FAILED, "failure": NodeStatus.FAILED, "invalid": NodeStatus.INVALIDATED}
    return aliases.get(text, NodeStatus(text) if text in {item.value for item in NodeStatus} else NodeStatus.SUCCESS)


def _add_rule_based_edges(graph: ExecutionGraph) -> None:
    ordered = sorted(graph.nodes.values(), key=lambda node: (node.timestamp, node.id))
    latest_by_type: Dict[NodeType, Node] = {}
    latest_error: Optional[Node] = None
    for node in ordered:
        if node.type == NodeType.PLAN:
            for req in [n for n in ordered if n.type == NodeType.REQUEST and n.timestamp <= node.timestamp]:
                graph.add_edge(req.id, node.id, "request_constrains_plan")
        elif node.type == NodeType.SERVICE_SELECTION and latest_by_type.get(NodeType.PLAN):
            graph.add_edge(latest_by_type[NodeType.PLAN].id, node.id, "plan_constrains_api")
        elif node.type == NodeType.PARAMETER_BINDING and latest_by_type.get(NodeType.SERVICE_SELECTION):
            graph.add_edge(latest_by_type[NodeType.SERVICE_SELECTION].id, node.id, "api_constrains_params")
        elif node.type == NodeType.API_INVOCATION and latest_by_type.get(NodeType.PARAMETER_BINDING):
            graph.add_edge(latest_by_type[NodeType.PARAMETER_BINDING].id, node.id, "params_trigger_call")
        elif node.type == NodeType.OBSERVATION and latest_by_type.get(NodeType.API_INVOCATION):
            graph.add_edge(latest_by_type[NodeType.API_INVOCATION].id, node.id, "call_produces_observation")
        elif node.type == NodeType.CLAIM:
            for obs in [n for n in ordered if n.type == NodeType.OBSERVATION and n.timestamp <= node.timestamp]:
                graph.add_edge(obs.id, node.id, "observation_supports_claim", bool(node.meta.get("supported_by") or node.meta.get("evidence")))
        elif node.type == NodeType.REPAIR and latest_error:
            graph.add_edge(latest_error.id, node.id, "error_triggers_repair")
        if node.status == NodeStatus.FAILED or node.meta.get("error"):
            latest_error = node
        latest_by_type[node.type] = node
