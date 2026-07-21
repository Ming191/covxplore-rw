from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from covxplore.hybrid.models import Route


FEATURE_NAMES = [
    "bias",
    "path_depth",
    "loop_depth",
    "constraint_count",
    "is_branch",
    "scalar_count",
    "string_count",
    "pointer_count",
    "container_count",
    "object_count",
    "external_call_count",
    "nonlinear_constraint_count",
    "statement_coverage",
    "branch_coverage",
    "symbolic_success_rate",
    "hybrid_success_rate",
    "llm_success_rate",
    "remaining_wall_time",
    "remaining_llm_calls",
    "remaining_symbolic_attempts",
    "remaining_test_executions",
]


def calculate_reward(
    *,
    delta_statement_fraction: float,
    delta_branch_fraction: float,
    route_seconds: float,
    route_tokens: int,
    invalid_test: bool,
) -> float:
    return (
        0.4 * delta_statement_fraction
        + 0.6 * delta_branch_fraction
        - 0.05 * min(max(route_seconds, 0.0) / 60.0, 1.0)
        - 0.05 * min(max(route_tokens, 0) / 8000.0, 1.0)
        - 0.20 * float(invalid_test)
    )


class Scheduler(Protocol):
    def choose(self, features: dict[str, float], available: set[Route]) -> Route: ...

    def update(self, route: Route, features: dict[str, float], reward: float) -> None: ...


@dataclass
class RuleScheduler:
    """Deterministic fallback used when no trained policy is available."""

    def choose(self, features: dict[str, float], available: set[Route]) -> Route:
        complex_cpp = any(
            features.get(name, 0.0) > 0
            for name in (
                "pointer_count",
                "container_count",
                "object_count",
                "external_call_count",
                "nonlinear_constraint_count",
            )
        )
        preferred = (
            [Route.HYBRID, Route.LLM, Route.SYMBOLIC]
            if complex_cpp
            else [Route.SYMBOLIC, Route.HYBRID, Route.LLM]
        )
        for route in preferred:
            if route in available:
                return route
        raise RuntimeError("no route is available")

    def update(self, route: Route, features: dict[str, float], reward: float) -> None:
        return None


def _identity(size: int, scale: float = 1.0) -> list[list[float]]:
    return [
        [scale if row == col else 0.0 for col in range(size)]
        for row in range(size)
    ]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b with pivoted Gauss-Jordan elimination."""

    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular LinUCB covariance matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            augmented[row] = [
                lhs - factor * rhs
                for lhs, rhs in zip(augmented[row], augmented[col])
            ]
    return [augmented[row][-1] for row in range(size)]


@dataclass
class LinUCBPolicy:
    alpha: float = 1.0
    ridge: float = 1.0
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    frozen: bool = False
    matrices: dict[Route, list[list[float]]] = field(init=False)
    rewards: dict[Route, list[float]] = field(init=False)
    observations: dict[Route, int] = field(init=False)

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        self.matrices = {
            route: _identity(size, self.ridge) for route in Route
        }
        self.rewards = {route: [0.0] * size for route in Route}
        self.observations = {route: 0 for route in Route}

    def _vector(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def choose(self, features: dict[str, float], available: set[Route]) -> Route:
        if not available:
            raise RuntimeError("no route is available")
        if all(self.observations[route] == 0 for route in available):
            return RuleScheduler().choose(features, available)
        x = self._vector(features)
        scores: dict[Route, float] = {}
        for route in sorted(available, key=lambda item: item.value):
            matrix = self.matrices[route]
            theta = _solve(matrix, self.rewards[route])
            inverse_x = _solve(matrix, x)
            mean = sum(lhs * rhs for lhs, rhs in zip(theta, x))
            uncertainty = math.sqrt(
                max(0.0, sum(lhs * rhs for lhs, rhs in zip(x, inverse_x)))
            )
            scores[route] = mean + self.alpha * uncertainty
        return max(scores, key=lambda route: (scores[route], route.value))

    def update(self, route: Route, features: dict[str, float], reward: float) -> None:
        if self.frozen:
            return
        x = self._vector(features)
        matrix = self.matrices[route]
        for row in range(len(x)):
            for col in range(len(x)):
                matrix[row][col] += x[row] * x[col]
            self.rewards[route][row] += reward * x[row]
        self.observations[route] += 1

    def to_dict(self) -> dict:
        return {
            "version": "1.0",
            "algorithm": "LinUCB",
            "alpha": self.alpha,
            "ridge": self.ridge,
            "featureNames": self.feature_names,
            "frozen": self.frozen,
            "matrices": {
                route.value: matrix for route, matrix in self.matrices.items()
            },
            "rewards": {
                route.value: values for route, values in self.rewards.items()
            },
            "observations": {
                route.value: value for route, value in self.observations.items()
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def freeze_to(self, path: Path) -> None:
        self.frozen = True
        self.save(path)

    @classmethod
    def load(cls, path: Path, *, frozen: bool | None = None) -> "LinUCBPolicy":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != "1.0" or data.get("algorithm") != "LinUCB":
            raise ValueError("unsupported policy file")
        policy = cls(
            alpha=float(data["alpha"]),
            ridge=float(data["ridge"]),
            feature_names=list(data["featureNames"]),
            frozen=bool(data.get("frozen", False) if frozen is None else frozen),
        )
        policy.matrices = {
            Route(name): [[float(value) for value in row] for row in matrix]
            for name, matrix in data["matrices"].items()
        }
        policy.rewards = {
            Route(name): [float(value) for value in values]
            for name, values in data["rewards"].items()
        }
        policy.observations = {
            Route(name): int(value)
            for name, value in data.get("observations", {}).items()
        }
        for route in Route:
            policy.observations.setdefault(route, 0)
        return policy
