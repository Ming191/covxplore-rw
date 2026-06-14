from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import litellm
from rich.console import Console

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.config import get_settings
from covxplore.coverage_models import TestResult
from covxplore.experiment import ExperimentConfig
from covxplore.prompts.builder import PromptBuilder
from covxplore.status import TestStatus
from covxplore.test_suite import TestSuite
from covxplore.tools.execute_testcase import test_result_from_execute_result

_console = Console()


@dataclass
class Candidate:
    test_name: str | None
    test_body: str
    target_node: int | None = None
    target_polarity: bool | None = None


class BatchGenerationRunner:
    def __init__(self, prompt_config, builder: PromptBuilder):
        self.prompt_config = prompt_config
        self.builder = builder
        cfg = get_settings()
        model = cfg.deepseek_model.strip()
        self.model = model if "/" in model else f"deepseek/{model}"
        self.api_key = cfg.deepseek_api_key
        self.base_url = cfg.deepseek_base_url
        self.max_tokens = cfg.max_tokens

    def run(
        self,
        config: ExperimentConfig,
        suite: TestSuite,
        *,
        static_conditions_text: str,
        static_context_text: str,
        static_source_text: str,
    ) -> None:
        no_gain_batches = 0
        for _ in range(config.max_iterations):
            if self._coverage_target_reached(suite, config):
                return
            if no_gain_batches >= config.redundant_streak_limit:
                return

            candidates = self._generate_candidates(
                config,
                suite,
                static_conditions_text=static_conditions_text,
                static_context_text=static_context_text,
                static_source_text=static_source_text,
            )
            if not candidates:
                _console.print("[yellow]Batch LLM returned no valid candidates.[/]")
                return

            kept = 0
            for candidate in candidates:
                if self._coverage_target_reached(suite, config):
                    return
                result = self._execute_candidate(config.function_path, candidate)
                gained = self._preview_gain(suite, result)
                if gained > 0 or len(suite.tests) < get_settings().min_suite_size:
                    suite.add_result(result, get_settings().min_suite_size)
                    kept += 1
                    _console.print(
                        f"[dim]batch kept {result.test_name}: {result.status}, "
                        f"+stmt={result.new_statements_covered}, "
                        f"+branch={result.new_branches_covered}, "
                        f"+mcdc={result.new_mcdc_pairs_covered}[/]"
                    )
                else:
                    _console.print(
                        f"[dim]batch skipped no-gain {result.test_name}: {result.status}[/]"
                    )
            if kept == 0:
                no_gain_batches += 1
            else:
                no_gain_batches = 0

    def _generate_candidates(
        self,
        config: ExperimentConfig,
        suite: TestSuite,
        *,
        static_conditions_text: str,
        static_context_text: str,
        static_source_text: str,
    ) -> list[Candidate]:
        task = self.builder.task_description(
            function_path=config.function_path,
            suite=suite,
            remaining_iterations=config.max_iterations - suite.iteration_count,
            static_conditions_text=static_conditions_text,
            static_context_text=static_context_text,
            static_source_text=static_source_text,
        )
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task},
        ]
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.base_url,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        return self._parse_candidates(content)

    def _system_prompt(self) -> str:
        base = self.builder.system_prompt()
        strict = (
            "Return only a JSON object with this exact shape: "
            '{"candidates":[{"test_name":"name","test_body":"C++ body",'
            '"target_node":123,"target_polarity":true}]}. '
            "Generate 3 to 5 diverse candidates. Do not call tools. "
            "Do not include markdown fences, prose, comments outside JSON, or main()."
        )
        return f"{base}\n\n{strict}" if base else strict

    def _parse_candidates(self, content: str) -> list[Candidate]:
        data = json.loads(_extract_json_object(content))
        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list):
            return []
        candidates: list[Candidate] = []
        for raw in raw_candidates[:5]:
            if not isinstance(raw, dict):
                continue
            test_body = raw.get("test_body")
            if not isinstance(test_body, str) or not test_body.strip():
                continue
            test_name = raw.get("test_name")
            candidates.append(
                Candidate(
                    test_name=test_name if isinstance(test_name, str) else None,
                    test_body=test_body.strip(),
                    target_node=_optional_int(raw.get("target_node")),
                    target_polarity=_optional_bool(raw.get("target_polarity")),
                )
            )
        return candidates

    def _preview_gain(self, suite: TestSuite, result: TestResult) -> int:
        preview = copy.deepcopy(suite)
        preview_result = result.model_copy(deep=True)
        preview.add_result(preview_result, get_settings().min_suite_size)
        return (
            preview_result.new_statements_covered
            + preview_result.new_branches_covered
            + preview_result.new_mcdc_pairs_covered
        )

    def _execute_candidate(self, absolute_path: str, candidate: Candidate) -> TestResult:
        t0 = time.monotonic()
        try:
            with AkaUTClient() as client:
                raw = client.execute_testcase(
                    absolute_path,
                    candidate.test_body,
                    candidate.test_name,
                )
            elapsed = (time.monotonic() - t0) * 1000
            return test_result_from_execute_result(raw, candidate.test_body, elapsed)
        except AkaUTError as exc:
            elapsed = (time.monotonic() - t0) * 1000
            return TestResult(
                test_name=candidate.test_name or "unknown",
                test_body=candidate.test_body,
                status=TestStatus.COMPILE_ERROR.value,
                execute_log=str(exc),
                elapsed_ms=elapsed,
            )

    @staticmethod
    def _coverage_target_reached(suite: TestSuite, config: ExperimentConfig) -> bool:
        mcdc_done = (
            suite.total_mcdc_conditions == 0
            or suite.mcdc_coverage_pct >= config.mcdc_target
            or not suite.unvisited_summary()
        )
        stmt_done = (
            suite.total_statements == 0
            or suite.covered_statements >= suite.total_statements
        )
        branch_done = (
            suite.total_branches == 0 or suite.covered_branches >= suite.total_branches
        )
        return suite.iteration_count > 0 and mcdc_done and stmt_done and branch_done


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return match.group(0)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None
