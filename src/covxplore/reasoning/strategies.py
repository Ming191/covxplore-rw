from __future__ import annotations

import ast
import json
import resource
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from crewai import LLM
from pydantic import BaseModel

from covxplore.agents.schemas import GenerateTestBatchAction
from covxplore.llm import build_llm
from covxplore.prompts.config import ReasoningTechnique
from covxplore.reasoning.schemas import (
    ChainOfThoughtResult,
    Decomposition,
    ProgramOfThoughts,
    ProgramScenarios,
    SubproblemSolution,
    ThoughtEvaluation,
    ThoughtPlan,
    ThoughtPlans,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ReasoningInput:
    system_prompt: str
    task_prompt: str


class ReasoningStrategy(ABC):
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or build_llm()

    @abstractmethod
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        """Generate one test batch without executing it."""

    def _json_call(self, prompt: str, schema: type[T]) -> T:
        messages = [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ]
        error = ""
        for _ in range(2):
            response = self.llm.call(messages)
            try:
                return schema.model_validate_json(_json_object(str(response)))
            except Exception as exc:
                error = str(exc)
                messages.extend(
                    [
                        {"role": "assistant", "content": str(response)},
                        {
                            "role": "user",
                            "content": f"Invalid output: {error}. Return a corrected JSON object only.",
                        },
                    ]
                )
        raise ValueError(f"LLM did not return valid {schema.__name__}: {error}")

    @staticmethod
    def _batch_prompt(reasoning_input: ReasoningInput, evidence: str = "") -> str:
        suffix = f"\n\nREASONING EVIDENCE:\n{evidence}" if evidence else ""
        return (
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}{suffix}\n\n"
            "Return JSON matching: "
            '{"candidates":[{"test_name":"...","test_body":"..."}]}.'
        )


class DirectStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        return self._json_call(self._batch_prompt(reasoning_input), GenerateTestBatchAction)


class ChainOfThoughtStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        prompt = (
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
            "Work step by step before committing to tests. Trace relevant conditions, derive concrete "
            "input/state constraints, check candidate diversity, then provide the final batch. Return JSON "
            'matching {"reasoning":"explicit step-by-step rationale",'
            '"candidates":[{"test_name":"...","test_body":"..."}]}.'
        )
        result = self._json_call(prompt, ChainOfThoughtResult)
        return GenerateTestBatchAction(candidates=result.candidates)


class LeastToMostStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        decomposition = self._json_call(
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
            "Decompose the test-generation problem into an ordered list of simpler prerequisite "
            "subproblems, from easiest setup/condition to hardest interaction. Do not solve them yet. "
            'Return {"subproblems":["..."]}.',
            Decomposition,
        )
        solved: list[dict] = []
        for subproblem in decomposition.subproblems:
            solution = self._json_call(
                f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
                f"PREVIOUSLY SOLVED SUBPROBLEMS:\n{json.dumps(solved, ensure_ascii=False)}\n\n"
                f"SOLVE NEXT SUBPROBLEM:\n{subproblem}\n\n"
                "Use prior solutions as established facts. Derive concrete C++ input/state scenarios. "
                'Return {"analysis":"...","scenarios":["..."]}.',
                SubproblemSolution,
            )
            solved.append({"subproblem": subproblem, **solution.model_dump()})
        return self._json_call(
            self._batch_prompt(reasoning_input, json.dumps(solved, ensure_ascii=False)),
            GenerateTestBatchAction,
        )


class TreeOfThoughtsStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        plans = self._json_call(
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
            "Propose 3-5 genuinely different batch-level plans. Each plan must target a different "
            "combination of coverage gaps or setup strategy. Do not write tests yet. Return "
            '{"plans":[{"plan_id":"p1","approach":"...","target_gaps":["..."],'
            '"setup_strategy":"..."}]}.',
            ThoughtPlans,
        )
        first_scores = self._evaluate(reasoning_input, plans)
        finalists = _rank_plans(plans, first_scores)[:2]
        expanded = [self._expand(reasoning_input, plan) for plan in finalists]
        expanded_plans = ThoughtPlans(plans=expanded + finalists[:1])
        final_scores = self._evaluate(reasoning_input, expanded_plans)
        winner = _rank_plans(expanded_plans, final_scores)[0]
        return self._json_call(
            self._batch_prompt(reasoning_input, winner.model_dump_json()),
            GenerateTestBatchAction,
        )

    def _evaluate(
        self, reasoning_input: ReasoningInput, plans: ThoughtPlans
    ) -> ThoughtEvaluation:
        return self._json_call(
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
            f"CANDIDATE PLANS:\n{plans.model_dump_json()}\n\n"
            "Evaluate every plan for expected new branch/statement coverage, feasibility of concrete "
            "C++ setup, compile risk, and overlap between proposed candidates. Score 0-10. Return "
            '{"scores":[{"plan_id":"p1","score":8,"rationale":"..."}]}.',
            ThoughtEvaluation,
        )

    def _expand(self, reasoning_input: ReasoningInput, plan: ThoughtPlan) -> ThoughtPlan:
        return self._json_call(
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
            f"PLAN TO EXPAND:\n{plan.model_dump_json()}\n\n"
            "Refine this plan after one-step lookahead: identify likely missed short-circuit/loop/error "
            "branches and improve its setup strategy. Return one plan with a new plan_id matching "
            '{"plan_id":"...","approach":"...","target_gaps":["..."],'
            '"setup_strategy":"..."}.',
            ThoughtPlan,
        )


class ProgramOfThoughtsStrategy(ReasoningStrategy):
    def generate_batch(self, reasoning_input: ReasoningInput) -> GenerateTestBatchAction:
        program = self._json_call(
            f"{reasoning_input.system_prompt}\n\n{reasoning_input.task_prompt}\n\n"
            "Write a small Python program that computes diverse concrete input/state scenarios from "
            "the focal predicates. The program must assign a JSON-serializable list of dictionaries to "
            "a variable named scenarios. It may use assignments, for/range, if, arithmetic, comparisons, "
            "and list.append only. No imports, files, network, classes, functions, while, exceptions, or "
            'dunder names. Return {"program":"..."}.',
            ProgramOfThoughts,
        )
        scenarios = execute_program_of_thoughts(program.program)
        return self._json_call(
            self._batch_prompt(reasoning_input, scenarios.model_dump_json()),
            GenerateTestBatchAction,
        )


_STRATEGIES: dict[ReasoningTechnique, type[ReasoningStrategy]] = {
    ReasoningTechnique.NONE: DirectStrategy,
    ReasoningTechnique.COT: ChainOfThoughtStrategy,
    ReasoningTechnique.LEAST_TO_MOST: LeastToMostStrategy,
    ReasoningTechnique.TREE_OF_THOUGHTS: TreeOfThoughtsStrategy,
    ReasoningTechnique.PROGRAM_OF_THOUGHTS: ProgramOfThoughtsStrategy,
}


def get_reasoning_strategy(
    technique: ReasoningTechnique, llm: LLM | None = None
) -> ReasoningStrategy:
    return _STRATEGIES[technique](llm)


def execute_program_of_thoughts(program: str) -> ProgramScenarios:
    tree = ast.parse(program, mode="exec")
    _ProgramValidator().visit(tree)
    wrapper = (
        "import json\n"
        + program
        + "\nprint(json.dumps(scenarios, ensure_ascii=False, separators=(',', ':')))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", wrapper],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
        preexec_fn=_restrict_process,
    )
    if len(completed.stdout) > 1_000_000:
        raise ValueError("Program-of-Thoughts output exceeds 1 MB")
    return ProgramScenarios(scenarios=json.loads(completed.stdout))


def _json_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return text[start : end + 1]


def _rank_plans(plans: ThoughtPlans, evaluation: ThoughtEvaluation) -> list[ThoughtPlan]:
    scores = {score.plan_id: score.score for score in evaluation.scores}
    missing = [plan.plan_id for plan in plans.plans if plan.plan_id not in scores]
    if missing:
        raise ValueError(f"Evaluator omitted plans: {', '.join(missing)}")
    return sorted(plans.plans, key=lambda plan: scores[plan.plan_id], reverse=True)


def _restrict_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


class _ProgramValidator(ast.NodeVisitor):
    _ALLOWED = (
        ast.Module,
        ast.Assign,
        ast.Expr,
        ast.For,
        ast.If,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.List,
        ast.Dict,
        ast.Tuple,
        ast.Subscript,
        ast.Slice,
        ast.Call,
        ast.Attribute,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
    )
    _CALLS = {"range", "len", "str", "int", "float", "bool"}

    def __init__(self) -> None:
        self.nodes = 0

    def generic_visit(self, node) -> None:
        self.nodes += 1
        if self.nodes > 500:
            raise ValueError("Program-of-Thoughts program is too large")
        if not isinstance(node, self._ALLOWED):
            raise ValueError(f"Program-of-Thoughts forbids {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise ValueError("Program-of-Thoughts forbids private names")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr != "append" or not isinstance(node.value, ast.Name):
            raise ValueError("Program-of-Thoughts only permits list.append")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        valid_name = isinstance(node.func, ast.Name) and node.func.id in self._CALLS
        valid_append = isinstance(node.func, ast.Attribute) and node.func.attr == "append"
        if not (valid_name or valid_append) or node.keywords:
            raise ValueError("Program-of-Thoughts call is not permitted")
        self.generic_visit(node)
