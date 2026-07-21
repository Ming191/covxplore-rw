from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.hybrid.artifacts import (
    append_summary,
    configuration_hash,
    result_row,
    write_artifacts,
)
from covxplore.hybrid.runner import HybridGenerationConfig, HybridGenerationRunner


@dataclass
class BatchConfig:
    generation: HybridGenerationConfig
    out_dir: Path
    function_paths: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    resume: bool = True


class HybridBatchRunner:
    """Sequential batch runner; AkaUT execution is intentionally single-process."""

    def __init__(self, client: AkaUTClient) -> None:
        self.client = client

    def discover(self, source_files: list[str]) -> list[str]:
        requested = [self._normalize(item) for item in source_files if item.strip()]
        nodes = self.client.search_nodes("", ["FUNCTION"])
        paths = []
        for node in nodes:
            normalized = self._normalize(node.absolute_path)
            if requested and not any(self._matches_source(normalized, item) for item in requested):
                continue
            paths.append(node.absolute_path)
        return sorted(set(paths), key=self._normalize)

    def run(self, config: BatchConfig) -> Path:
        paths = config.function_paths or self.discover(config.source_files)
        if not paths:
            raise ValueError("batch contains no function paths")
        config.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = config.out_dir / "batch_summary.csv"
        completed = self._completed(summary_path) if config.resume else set()

        for index, function_path in enumerate(paths, start=1):
            generation = replace(config.generation, function_path=function_path)
            identity = (function_path, configuration_hash(generation))
            if identity in completed:
                continue
            run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{index:04d}"
            generation = replace(
                generation,
                run_id=run_id,
            )
            try:
                result = HybridGenerationRunner(client=self.client).run(generation)
                write_artifacts(result, config.out_dir / "runs")
                append_summary(summary_path, result_row(result))
            except Exception as error:
                row = self._error_row(generation, error)
                append_summary(summary_path, row)
        return summary_path

    @staticmethod
    def load_paths(path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    @staticmethod
    def _completed(path: Path) -> set[tuple[str, str]]:
        if not path.is_file():
            return set()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return {
                (row.get("function_path", ""), row.get("config_hash", ""))
                for row in csv.DictReader(stream)
                if row.get("function_path") and row.get("config_hash")
                and row.get("stop_reason") not in {"error", "infra_error"}
            }

    @staticmethod
    def _error_row(config: HybridGenerationConfig, error: Exception) -> dict[str, object]:
        return {
            "function_name": Path(config.function_path).name,
            "function_path": config.function_path,
            "run_id": config.run_id,
            "config_hash": configuration_hash(config),
            "strategy": config.strategy,
            "scheduler": config.scheduler_mode,
            "stop_reason": "infra_error" if isinstance(error, AkaUTError) else "error",
            "error": f"{type(error).__name__}: {error}",
        }

    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace("\\", "/").strip().lower()

    @classmethod
    def _matches_source(cls, function_path: str, requested: str) -> bool:
        requested_name = requested.rsplit("/", 1)[-1]
        marker = re.search(r"\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)(?=/)", function_path)
        source_prefix = function_path[: marker.end()] if marker else function_path
        return (
            source_prefix == requested
            or source_prefix.endswith("/" + requested_name)
            or f"/{requested_name}/" in function_path
        )
