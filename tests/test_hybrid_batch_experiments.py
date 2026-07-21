from __future__ import annotations

import csv
from pathlib import Path

import pytest

from covxplore.api_client import NodeInfo
from covxplore.hybrid.artifacts import configuration_hash
from covxplore.hybrid.batch import HybridBatchRunner
from covxplore.hybrid.experiments import HybridExperimentRunner
from covxplore.hybrid.runner import HybridGenerationConfig


class SearchClient:
    def search_nodes(self, query: str, types: list[str]):
        assert query == ""
        assert types == ["FUNCTION"]
        return [
            NodeInfo(
                name="f",
                qualified_name="ns::f()",
                absolute_path="D:\\p\\src\\hjson_decode.cpp\\ns::f()",
                type="FUNCTION",
                line=10,
            ),
            NodeInfo(
                name="g",
                qualified_name="ns::g()",
                absolute_path="D:\\p\\src\\other.cpp\\ns::g()",
                type="FUNCTION",
                line=20,
            ),
        ]


def test_batch_discovers_functions_by_windows_source_name() -> None:
    paths = HybridBatchRunner(SearchClient()).discover(["hjson_decode.cpp"])
    assert paths == ["D:\\p\\src\\hjson_decode.cpp\\ns::f()"]
    assert HybridBatchRunner._matches_source(
        "d:/p/src/parser.cc/ns::parse()", "parser.cc"
    )


def test_batch_paths_file_and_resume_identity_use_full_configuration(
    tmp_path: Path,
) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("# comment\nD:/p/a.cpp/f()\n\n", encoding="utf-8")
    assert HybridBatchRunner.load_paths(paths_file) == ["D:/p/a.cpp/f()"]

    first = HybridGenerationConfig(function_path="D:/p/a.cpp/f()")
    changed = HybridGenerationConfig(
        function_path="D:/p/a.cpp/f()", branch_target=0.75
    )
    assert configuration_hash(first) != configuration_hash(changed)

    summary = tmp_path / "batch_summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["function_path", "config_hash", "stop_reason"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "function_path": first.function_path,
                "config_hash": configuration_hash(first),
                "stop_reason": "coverage_target",
            }
        )
        writer.writerow(
            {
                "function_path": changed.function_path,
                "config_hash": configuration_hash(changed),
                "stop_reason": "infra_error",
            }
        )
    assert HybridBatchRunner._completed(summary) == {
        (first.function_path, configuration_hash(first))
    }


@pytest.mark.parametrize(
    ("variant", "strategy", "scheduler", "flag"),
    [
        ("llm", "llm", "rule", None),
        ("symbolic", "symbolic", "rule", None),
        ("hybrid_rule", "hybrid", "rule", None),
        ("no_nearest_seed", "hybrid", "rule", "use_nearest_seed"),
        (
            "no_first_divergence_repair",
            "hybrid",
            "rule",
            "use_divergence_repair",
        ),
        ("no_symbolic_partial", "hybrid", "rule", "use_symbolic_partial"),
    ],
)
def test_experiment_variants_map_to_runner_configuration(
    variant: str, strategy: str, scheduler: str, flag: str | None
) -> None:
    base = HybridGenerationConfig(function_path="D:/p/a.cpp/f()")
    configured = HybridExperimentRunner(client=None)._variant(base, variant)
    assert configured.strategy == strategy
    assert configured.scheduler_mode == scheduler
    if flag is not None:
        assert getattr(configured, flag) is False


def test_frozen_experiment_requires_policy() -> None:
    base = HybridGenerationConfig(function_path="D:/p/a.cpp/f()")
    with pytest.raises(ValueError, match="requires --policy"):
        HybridExperimentRunner(client=None)._variant(base, "hybrid_frozen")
