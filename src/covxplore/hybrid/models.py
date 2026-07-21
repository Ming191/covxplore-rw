from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel


CONTRACT_VERSION = "1.0"


class ContractModel(BaseModel):
    """Base for the Java/Python JSON contract."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
        use_enum_values=False,
    )

    def wire_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class TargetKind(str, Enum):
    STATEMENT = "STATEMENT"
    BRANCH_EDGE = "BRANCH_EDGE"


class ActionKind(str, Enum):
    DECLARE = "DECLARE"
    CONSTRUCT = "CONSTRUCT"
    ASSIGN = "ASSIGN"
    CALL = "CALL"
    STUB = "STUB"
    RAW_CPP = "RAW_CPP"
    INVOKE = "INVOKE"


class SolverStatus(str, Enum):
    SOLVED = "SOLVED"
    PARTIAL = "PARTIAL"
    UNSAT = "UNSAT"
    UNSUPPORTED = "UNSUPPORTED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class Route(str, Enum):
    SYMBOLIC = "SYMBOLIC"
    HYBRID = "HYBRID"
    LLM = "LLM"


class SourceRange(ContractModel):
    file: str
    line_in_function: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class Statement(ContractModel):
    key: str
    text: str
    source_range: SourceRange
    path_depth: int = 0
    loop_depth: int = 0
    expected_path: list[str] = Field(default_factory=list)


class BranchEdge(ContractModel):
    key: str
    source_statement_key: str
    destination_statement_key: str | None = None
    outcome: str
    label: str | None = None
    source_range: SourceRange
    path_depth: int = 0
    loop_depth: int = 0
    downstream_statement_keys: list[str] = Field(default_factory=list)
    expected_path: list[str] = Field(default_factory=list)


class FunctionFeatures(ContractModel):
    constraint_count: int = 0
    scalar_count: int = 0
    string_count: int = 0
    pointer_count: int = 0
    container_count: int = 0
    object_count: int = 0
    external_call_count: int = 0
    nonlinear_constraint_count: int = 0


class CoverageModel(ContractModel):
    version: Literal["1.0"] = CONTRACT_VERSION
    function_path: str
    function_name: str
    model_hash: str
    source: str = ""
    statements: list[Statement] = Field(default_factory=list)
    branch_edges: list[BranchEdge] = Field(default_factory=list)
    features: FunctionFeatures = Field(default_factory=FunctionFeatures)

    @property
    def statement_keys(self) -> set[str]:
        return {item.key for item in self.statements}

    @property
    def branch_keys(self) -> set[str]:
        return {item.key for item in self.branch_edges}


class Target(ContractModel):
    kind: TargetKind
    key: str
    desired_outcome: bool | None = None


class Seed(ContractModel):
    intent_id: str | None = None
    test_name: str | None = None
    visited_statement_keys: list[str] = Field(default_factory=list)
    visited_branch_keys: list[str] = Field(default_factory=list)
    ordered_trace: list["TraceStep"] = Field(default_factory=list)
    bindings: list["TestBinding"] = Field(default_factory=list)


class TestAction(ContractModel):
    id: str
    kind: ActionKind
    code: str
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("id", "code")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TestBinding(ContractModel):
    cpp_lvalue: str
    cpp_type: str | None = None
    value: str
    origin: str
    locked: bool = False

    @field_validator("cpp_lvalue")
    @classmethod
    def safe_cpp_lvalue(cls, value: str) -> str:
        stripped = value.strip()
        if not re.fullmatch(
            r"(?:::)?[A-Za-z_]\w*(?:(?:::|\.|->)[A-Za-z_]\w*|\[(?:0|[1-9]\d*)])*",
            stripped,
        ):
            raise ValueError("must be an assignable C++ lvalue without function calls")
        return stripped

    @field_validator("value", "origin")
    @classmethod
    def binding_text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TestIntent(ContractModel):
    version: Literal["1.0"] = CONTRACT_VERSION
    intent_id: str
    function_path: str
    target: Target
    seed: Seed | None = None
    actions: list[TestAction] = Field(default_factory=list)
    bindings: list[TestBinding] = Field(default_factory=list)
    expected_trace: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @property
    def is_executable(self) -> bool:
        return sum(action.kind == ActionKind.INVOKE for action in self.actions) == 1

    @model_validator(mode="after")
    def action_ids_are_unique(self) -> "TestIntent":
        binding_names = [binding.cpp_lvalue for binding in self.bindings]
        if len(binding_names) != len(set(binding_names)):
            raise ValueError("binding cppLvalue values must be unique")
        ids = [action.id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("action ids must be unique")
        known = set(ids)
        for action in self.actions:
            missing = set(action.depends_on) - known
            if missing:
                raise ValueError(
                    f"action {action.id!r} depends on unknown actions: {sorted(missing)}"
                )
        indegree = {action.id: len(action.depends_on) for action in self.actions}
        outgoing: dict[str, list[str]] = {action.id: [] for action in self.actions}
        for action in self.actions:
            for dependency in action.depends_on:
                outgoing[dependency].append(action.id)
        ready = [action_id for action_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            action_id = ready.pop()
            visited += 1
            for dependent in outgoing[action_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if visited != len(self.actions):
            raise ValueError("action dependency graph contains a cycle")
        return self

    def validate_executable(self) -> None:
        invoke_count = sum(
            action.kind == ActionKind.INVOKE for action in self.actions
        )
        if invoke_count != 1:
            raise ValueError(
                f"an executable TestIntent requires exactly one INVOKE action; got {invoke_count}"
            )


class SymbolicAttemptRequest(ContractModel):
    version: Literal["1.0"] = CONTRACT_VERSION
    function_path: str
    target: Target
    seed: Seed | None = None
    loop_bound: int = Field(default=3, ge=1, le=10)
    solver_timeout_sec: int = Field(default=60, ge=1, le=600)


class SymbolicAttemptResponse(ContractModel):
    version: Literal["1.0"] = CONTRACT_VERSION
    status: SolverStatus
    intent: TestIntent | None = None
    path_keys: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    solver_status: SolverStatus | None = None
    solver_calls: int = 0
    solver_status_counts: dict[str, int] = Field(default_factory=dict)
    path_candidate_count: int = 0
    path_planning_elapsed_ms: float = 0.0
    constraint_extraction_elapsed_ms: float = 0.0
    solver_elapsed_ms: float = 0.0
    model_binding_count: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


class ExecuteIntentRequest(ContractModel):
    version: Literal["1.0"] = CONTRACT_VERSION
    intent: TestIntent
    test_name: str | None = None


class CoverageDetail(ContractModel):
    visited: int = 0
    total: int = 0
    progress: float = 0.0


class TraceStep(ContractModel):
    index: int
    kind: TargetKind
    key: str
    outcome: str | None = None


class FirstDivergence(ContractModel):
    index: int
    expected_key: str | None = None
    actual_key: str | None = None
    reason: str


class ExecutionResult(ContractModel):
    version: Literal["1.0"] = CONTRACT_VERSION
    test_name: str
    status: str
    execute_log: str = ""
    generated_test_body: str = ""
    statement_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    branch_coverage: CoverageDetail = Field(default_factory=CoverageDetail)
    visited_statement_keys: list[str] = Field(default_factory=list)
    visited_branch_keys: list[str] = Field(default_factory=list)
    ordered_trace: list[TraceStep] = Field(default_factory=list)
    trace_truncated: bool = False
    target_reached: bool = False
    first_divergence: FirstDivergence | None = None
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status.upper() == "PASSED"


class RouteFeatures(ContractModel):
    values: dict[str, float] = Field(default_factory=dict)

    def vector(self, names: list[str]) -> list[float]:
        return [float(self.values.get(name, 0.0)) for name in names]


class RouteDecision(ContractModel):
    iteration: int
    target: Target
    route: Route
    features: dict[str, float]
    reward: float | None = None
    reason: str = ""


class TestAttemptRecord(ContractModel):
    iteration: int
    route: Route
    target: Target
    intent: TestIntent | None = None
    symbolic: SymbolicAttemptResponse | None = None
    execution: ExecutionResult | None = None
    new_statement_keys: list[str] = Field(default_factory=list)
    new_branch_keys: list[str] = Field(default_factory=list)
    redundant: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0
    run_elapsed_ms: float = 0.0
    cumulative_statement_cov: float = 0.0
    cumulative_branch_cov: float = 0.0
    error: str | None = None


Seed.model_rebuild()
