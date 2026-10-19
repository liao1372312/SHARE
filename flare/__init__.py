"""FLARE: fault localization and minimal repair for service composition traces."""

from .core import (
    Agent,
    Edge,
    Event,
    ExecutionGraph,
    FailureType,
    Node,
    NodeStatus,
    NodeType,
    RepairAction,
)
from .backbone import MultiAgentComposer, CompositionOutput
from .pipeline import FLARE, FLAREConfig, FLAREResult, RepairIteration
from .repair import execute_repair, invalidated_nodes, merge_repaired_slice
from .claims import add_claim_nodes, claim_nodes, fclaim, unsupported_claim_nodes
from .predicates import exec_err, fail, missing_dep, repair_succ, schema_err, unrecovered, unsupported_claim

__all__ = [
    "Agent",
    "Edge",
    "Event",
    "ExecutionGraph",
    "FailureType",
    "Node",
    "NodeStatus",
    "NodeType",
    "RepairAction",
    "FLARE",
    "FLAREConfig",
    "FLAREResult",
    "RepairIteration",
    "MultiAgentComposer",
    "CompositionOutput",
    "invalidated_nodes",
    "execute_repair",
    "merge_repaired_slice",
    "fclaim",
    "add_claim_nodes",
    "claim_nodes",
    "unsupported_claim_nodes",
    "fail",
    "exec_err",
    "schema_err",
    "missing_dep",
    "unsupported_claim",
    "unrecovered",
    "repair_succ",
]
