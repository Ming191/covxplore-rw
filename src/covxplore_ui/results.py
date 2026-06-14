from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from covxplore.events import keys_to_camel


def result_paths(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    return sorted(out_dir.rglob("gen_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_to_run_state(summary: dict[str, Any], result_path: Path | None = None) -> dict[str, Any]:
    metrics = keys_to_camel(summary.get("metrics") or {})
    tests = [
        _test_to_camel(test)
        for test in (summary.get("test_suite") or [])
    ]
    return {
        "runId": summary.get("run_id", ""),
        "status": "failed" if summary.get("error") else "completed",
        "functionPath": summary.get("function_path", ""),
        "variant": summary.get("prompt_variant", ""),
        "metrics": metrics,
        "tests": tests,
        "error": summary.get("error"),
        "resultPath": str(result_path) if result_path is not None else None,
        "stopReason": summary.get("stop_reason"),
    }


def _test_to_camel(test: dict[str, Any]) -> dict[str, Any]:
    data = keys_to_camel(test)
    data["isPassed"] = data.get("status") == "PASSED"
    return data


def find_result(out_dir: Path, run_id: str) -> tuple[Path, dict[str, Any]] | None:
    for path in result_paths(out_dir):
        try:
            summary = load_result(path)
        except Exception:
            continue
        if summary.get("run_id") == run_id:
            return path, summary
    return None
