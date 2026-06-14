from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from covxplore.events import keys_to_camel
from covxplore_ui.results import summary_to_run_state


def build_report_from_summary(
    summary: dict[str, Any],
    result_path: Path | None = None,
) -> dict[str, Any]:
    run = summary_to_run_state(summary, result_path)
    raw_tests = summary.get("test_suite") or []
    tests = [_normalize_test(test) for test in raw_tests]
    metrics = keys_to_camel(summary.get("metrics") or {})
    static_conditions = keys_to_camel(summary.get("static_conditions") or [])
    source_text = summary.get("function_source") or ""

    return _build_report(
        run=run,
        tests=tests,
        metrics=metrics,
        source_text=source_text,
        static_conditions=static_conditions,
        llm_interactions=keys_to_camel(summary.get("llm_interactions") or []),
        legacy=_is_legacy_result(raw_tests),
    )


def build_report_from_run_state(
    state: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = events or []
    tests = [_normalize_test(test) for test in state.get("tests") or []]
    static_conditions: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "static_prefetch_completed":
            payload = event.get("payload") or {}
            static_conditions = keys_to_camel(payload.get("conditions") or [])

    return _build_report(
        run=state,
        tests=tests,
        metrics=state.get("metrics") or {},
        source_text="",
        static_conditions=static_conditions,
        llm_interactions=[],
        legacy=False,
    )


def _build_report(
    *,
    run: dict[str, Any],
    tests: list[dict[str, Any]],
    metrics: dict[str, Any],
    source_text: str,
    static_conditions: list[dict[str, Any]],
    llm_interactions: list[dict[str, Any]],
    legacy: bool,
) -> dict[str, Any]:
    conditions = _condition_reports(static_conditions, tests)
    statements = _statement_reports(tests)
    branches = _branch_reports(tests)
    source = _source_report(source_text, conditions, statements, branches)
    status_counts = Counter(test.get("status") or "UNKNOWN" for test in tests)
    public_tests = [
        {key: value for key, value in test.items() if not key.startswith("_")}
        for test in tests
    ]

    return {
        "run": run,
        "summary": {
            "statusCounts": dict(status_counts),
            "elapsedSec": metrics.get("elapsedSec", 0),
            "totalInputTokens": metrics.get("totalInputTokens", 0),
            "totalOutputTokens": metrics.get("totalOutputTokens", 0),
            "totalTokens": metrics.get("totalTokens", 0),
            "stopReason": run.get("stopReason"),
            "legacy": legacy,
        },
        "source": source,
        "conditions": conditions,
        "statements": statements,
        "branches": branches,
        "tests": public_tests,
        "llmInteractions": llm_interactions,
    }


def _normalize_test(test: dict[str, Any]) -> dict[str, Any]:
    data = keys_to_camel(test)
    data["_hasUnvisitedStatements"] = "unvisitedStatements" in data
    data["_hasUnvisitedBranches"] = "unvisitedBranches" in data
    data.setdefault("conditionTrace", [])
    data.setdefault("unvisitedMcdc", [])
    data.setdefault("unvisitedStatements", [])
    data.setdefault("unvisitedBranches", [])
    data.setdefault("traceSummary", {})
    data["isPassed"] = data.get("status") == "PASSED"
    data["tokenInput"] = data.get("tokenInput", 0) or 0
    data["tokenOutput"] = data.get("tokenOutput", 0) or 0
    data["elapsedMs"] = data.get("elapsedMs", 0) or 0
    return data


def _condition_reports(
    static_conditions: list[dict[str, Any]],
    tests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}

    for item in static_conditions:
        node_id = _int_or_none(item.get("nodeId"))
        if node_id is None:
            continue
        by_id[node_id] = {
            "nodeId": node_id,
            "condition": item.get("condition", ""),
            "lineInFunction": item.get("lineInFunction"),
            "startOffset": item.get("startOffset"),
            "endOffset": item.get("endOffset"),
            "trueCovered": False,
            "falseCovered": False,
            "tests": [],
        }

    for test in tests:
        for entry in list(test.get("conditionTrace") or []) + list(
            test.get("unvisitedMcdc") or []
        ):
            node_id = _int_or_none(entry.get("nodeId"))
            if node_id is None:
                continue
            report = by_id.setdefault(
                node_id,
                {
                    "nodeId": node_id,
                    "condition": entry.get("condition", ""),
                    "lineInFunction": entry.get("lineInFunction"),
                    "startOffset": entry.get("startOffsetInFunction")
                    or entry.get("startOffset"),
                    "endOffset": entry.get("endOffsetInFunction")
                    or entry.get("endOffset"),
                    "trueCovered": False,
                    "falseCovered": False,
                    "tests": [],
                },
            )
            report["condition"] = report.get("condition") or entry.get("condition", "")
            if entry.get("lineInFunction") is not None:
                report["lineInFunction"] = entry.get("lineInFunction")
            if entry.get("trueBranchVisited"):
                report["trueCovered"] = True
                _append_test_ref(report["tests"], test, True)
            if entry.get("falseBranchVisited"):
                report["falseCovered"] = True
                _append_test_ref(report["tests"], test, False)

    reports = list(by_id.values())
    for report in reports:
        missing = []
        if not report["trueCovered"]:
            missing.append("TRUE")
        if not report["falseCovered"]:
            missing.append("FALSE")
        report["missing"] = missing
    return sorted(reports, key=lambda item: (item.get("nodeId") is None, item.get("nodeId")))


def _statement_reports(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    cumulative: set[int] | None = None

    for test in tests:
        if not test.get("_hasUnvisitedStatements"):
            continue
        items = test.get("unvisitedStatements") or []
        current: set[int] = set()
        for item in items:
            node_id = _int_or_none(item.get("nodeId"))
            if node_id is None:
                continue
            current.add(node_id)
            seen.setdefault(
                node_id,
                {
                    "nodeId": node_id,
                    "statement": item.get("statement", ""),
                    "lineInFunction": item.get("lineInFunction"),
                    "startOffset": item.get("startOffset"),
                    "endOffset": item.get("endOffset"),
                },
            )
        cumulative = current if cumulative is None else cumulative & current

    cumulative = cumulative or set()
    reports = []
    for node_id, item in seen.items():
        reports.append({**item, "covered": node_id not in cumulative})
    return sorted(reports, key=lambda item: (item.get("lineInFunction") is None, item.get("lineInFunction") or 0))


def _branch_reports(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    cumulative_missing: set[tuple[int, bool]] | None = None
    for test in tests:
        if not test.get("_hasUnvisitedBranches"):
            continue
        current_missing: set[tuple[int, bool]] = set()
        for item in test.get("unvisitedBranches") or []:
            node_id = _int_or_none(item.get("nodeId"))
            if node_id is None:
                continue
            report = by_id.setdefault(
                node_id,
                {
                    "nodeId": node_id,
                    "condition": item.get("condition", ""),
                    "lineInFunction": item.get("lineInFunction"),
                    "startOffset": item.get("startOffset"),
                    "endOffset": item.get("endOffset"),
                    "trueCovered": False,
                    "falseCovered": False,
                },
            )
            if not item.get("trueVisited"):
                current_missing.add((node_id, True))
            if not item.get("falseVisited"):
                current_missing.add((node_id, False))

        cumulative_missing = (
            current_missing
            if cumulative_missing is None
            else cumulative_missing & current_missing
        )

    cumulative_missing = cumulative_missing or set()
    reports = list(by_id.values())
    for report in reports:
        node_id = report["nodeId"]
        report["trueCovered"] = (node_id, True) not in cumulative_missing
        report["falseCovered"] = (node_id, False) not in cumulative_missing
        missing = []
        if not report["trueCovered"]:
            missing.append("TRUE")
        if not report["falseCovered"]:
            missing.append("FALSE")
        report["missing"] = missing
    return sorted(reports, key=lambda item: (item.get("lineInFunction") is None, item.get("lineInFunction") or 0))


def _source_report(
    source_text: str,
    conditions: list[dict[str, Any]],
    statements: list[dict[str, Any]],
    branches: list[dict[str, Any]],
) -> dict[str, Any]:
    lines = source_text.splitlines()
    statement_by_line: dict[int, str] = {}
    condition_by_line: dict[int, list[int]] = {}
    branch_by_line: dict[int, list[int]] = {}

    for statement in statements:
        line = _int_or_none(statement.get("lineInFunction"))
        if line is not None:
            statement_by_line[line] = "covered" if statement.get("covered") else "uncovered"
    for condition in conditions:
        line = _int_or_none(condition.get("lineInFunction"))
        node_id = _int_or_none(condition.get("nodeId"))
        if line is not None and node_id is not None:
            condition_by_line.setdefault(line, []).append(node_id)
    for branch in branches:
        line = _int_or_none(branch.get("lineInFunction"))
        node_id = _int_or_none(branch.get("nodeId"))
        if line is not None and node_id is not None:
            branch_by_line.setdefault(line, []).append(node_id)

    return {
        "lines": [
            {
                "lineNumber": idx,
                "displayLineNumber": idx + 1,
                "text": text,
                "statementStatus": statement_by_line.get(idx, "unknown"),
                "conditionIds": condition_by_line.get(idx, []),
                "branchIds": branch_by_line.get(idx, []),
            }
            for idx, text in enumerate(lines)
        ]
    }


def _append_test_ref(target: list[dict[str, Any]], test: dict[str, Any], polarity: bool) -> None:
    ref = {
        "iteration": test.get("iteration"),
        "testName": test.get("testName"),
        "polarity": "TRUE" if polarity else "FALSE",
    }
    if ref not in target:
        target.append(ref)


def _has_trace_details(tests: list[dict[str, Any]]) -> bool:
    return any(
        test.get("conditionTrace")
        or test.get("unvisitedMcdc")
        or test.get("unvisitedStatements")
        or test.get("unvisitedBranches")
        for test in tests
    )


def _is_legacy_result(raw_tests: list[dict[str, Any]]) -> bool:
    if not raw_tests:
        return False
    trace_keys = {
        "condition_trace",
        "conditionTrace",
        "unvisited_mcdc",
        "unvisitedMcdc",
        "unvisited_statements",
        "unvisitedStatements",
        "unvisited_branches",
        "unvisitedBranches",
        "trace_summary",
        "traceSummary",
    }
    return not any(trace_keys.intersection(test.keys()) for test in raw_tests)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
