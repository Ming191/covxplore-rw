from __future__ import annotations

import csv
import json
from pathlib import Path

from covxplore.hybrid.artifacts import (
    SUMMARY_COLUMNS,
    append_summary,
    write_artifacts,
)
from covxplore.hybrid.budget import RunBudget
from covxplore.hybrid.coverage import ExactCoverageState
from covxplore.hybrid.models import (
    BranchEdge,
    CoverageModel,
    ExecutionResult,
    Route,
    SolverStatus,
    SourceRange,
    Statement,
    SymbolicAttemptResponse,
    Target,
    TargetKind,
    TestAttemptRecord,
)
from covxplore.hybrid.runner import HybridGenerationConfig, HybridGenerationResult


def test_write_artifacts_has_statement_branch_schema_without_mcdc(tmp_path: Path) -> None:
    model = CoverageModel(
        function_path="D:/src/f.cpp/f()",
        function_name="f()",
        model_hash="hash",
        statements=[
            Statement(
                key="s1",
                text="return;",
                source_range=SourceRange(file="D:/src/f.cpp"),
            )
        ],
    )
    state = ExactCoverageState(model, covered_statement_keys={"s1"})
    budget = RunBudget()
    budget.finish()
    result = HybridGenerationResult(
        config=HybridGenerationConfig(function_path=model.function_path),
        coverage_model=model,
        state=state,
        stop_reason="coverage_target",
        attempts=[],
        route_decisions=[],
        llm_interactions=[],
        budget=budget,
    )

    json_path, csv_path, summary_path = write_artifacts(result, tmp_path)
    write_artifacts(result, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"][
        "statement_cov"
    ] == 1.0
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert list(row) == SUMMARY_COLUMNS
    assert row["statement_cov"] == "1.0"
    assert row["branch_cov"] == "1.0"
    assert not any("mcdc" in column.lower() for column in row)
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_summary_reports_z3_and_symbolic_assistance_separately() -> None:
    source_range = SourceRange(file="D:/src/f.cpp")
    model = CoverageModel(
        function_path="D:/src/f.cpp/f()",
        function_name="f()",
        model_hash="hash",
        statements=[
            Statement(key="s1", text="if (x)", source_range=source_range),
            Statement(key="s2", text="return;", source_range=source_range),
        ],
        branch_edges=[
            BranchEdge(
                key="b1",
                source_statement_key="s1",
                outcome="TRUE",
                source_range=source_range,
            )
        ],
    )
    target = Target(kind=TargetKind.BRANCH_EDGE, key="b1", desired_outcome=True)
    hybrid = TestAttemptRecord(
        iteration=1,
        route=Route.HYBRID,
        target=target,
        symbolic=SymbolicAttemptResponse(
            status=SolverStatus.PARTIAL,
            solver_status=SolverStatus.SOLVED,
            solver_calls=2,
            solver_status_counts={"UNSAT": 1, "SOLVED": 1},
            solver_elapsed_ms=30_000,
            elapsed_ms=31_000,
            model_binding_count=2,
        ),
        execution=ExecutionResult(
            test_name="hybrid",
            status="PASSED",
            target_reached=True,
        ),
        new_statement_keys=["s1"],
        new_branch_keys=["b1"],
    )
    direct = TestAttemptRecord(
        iteration=2,
        route=Route.SYMBOLIC,
        target=Target(kind=TargetKind.STATEMENT, key="s2"),
        symbolic=SymbolicAttemptResponse(
            status=SolverStatus.SOLVED,
            solver_status=SolverStatus.SOLVED,
            solver_calls=1,
            solver_status_counts={"SOLVED": 1},
            solver_elapsed_ms=5_000,
            elapsed_ms=6_000,
            model_binding_count=1,
        ),
        execution=ExecutionResult(
            test_name="symbolic",
            status="PASSED",
            target_reached=True,
        ),
        new_statement_keys=["s2"],
    )
    budget = RunBudget()
    budget.finish()
    result = HybridGenerationResult(
        config=HybridGenerationConfig(function_path=model.function_path),
        coverage_model=model,
        state=ExactCoverageState(
            model,
            covered_statement_keys={"s1", "s2"},
            covered_branch_keys={"b1"},
        ),
        stop_reason="coverage_target",
        attempts=[hybrid, direct],
        route_decisions=[],
        llm_interactions=[],
        budget=budget,
    )

    metrics = result.to_summary_dict()["metrics"]

    assert metrics["z3_calls"] == 3
    assert metrics["z3_solved"] == 2
    assert metrics["z3_unsat"] == 1
    assert metrics["z3_solve_rate"] == 2 / 3
    assert metrics["z3_time_used"] == 35.0
    assert metrics["symbolic_time_used"] == 37.0
    assert metrics["z3_binding_count"] == 3
    assert metrics["symbolic_assisted_tests"] == 2
    assert metrics["symbolic_useful_rate"] == 1.0
    assert metrics["symbolic_assisted_new_statements"] == 2
    assert metrics["symbolic_assisted_new_branches"] == 1
    assert metrics["symbolic_new_statements"] == 1
    assert metrics["hybrid_assisted_new_statements"] == 1
    assert metrics["hybrid_assisted_new_branches"] == 1
    assert metrics["partial_to_passing_rate"] == 1.0


def test_append_summary_migrates_legacy_csv_without_losing_rows(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "batch_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "function_name",
                "function_path",
                "config_hash",
                "stop_reason",
                "total_token",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "function_name": "old()",
                "function_path": "D:/old.cpp/old()",
                "config_hash": "old-hash",
                "stop_reason": "coverage_target",
                "total_token": 123,
            }
        )

    append_summary(
        summary_path,
        {
            "function_name": "new()",
            "function_path": "D:/new.cpp/new()",
            "config_hash": "new-hash",
            "stop_reason": "routes_exhausted",
            "z3_calls": 2,
        },
    )

    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == SUMMARY_COLUMNS
    assert len(rows) == 2
    assert rows[0]["function_name"] == "old()"
    assert rows[0]["config_hash"] == "old-hash"
    assert rows[1]["function_name"] == "new()"
    assert rows[1]["z3_calls"] == "2"
    assert "total_token" not in rows[0]
    assert not summary_path.with_name("batch_summary.csv.tmp").exists()
