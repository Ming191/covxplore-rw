"""Constraint solver protocols — decouple Z3 from prompt generation.

Solvers produce human-readable hints describing what input values satisfy
each branch of a condition.  These hints are injected into the agent's
static conditions block so the LLM can skip arithmetic reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ConditionHint:
    """Z3-derived guidance for satisfying one condition's TRUE/FALSE branches.

    ``is_solvable=False`` means the solver cannot help (compound, pointer, or
    unsupported type).  ``true_branch`` / ``false_branch`` are C++-ready
    snippets like ``"p->indexNext >= 2"``.
    """

    node_id: int | None
    is_solvable: bool
    true_branch: str
    false_branch: str


class ConstraintSolver(Protocol):
    """Resolve input constraints for a single condition.

    Takes the raw condition text and a CDT-resolved variable→type map.
    Returns a ``ConditionHint`` or ``None`` when the condition cannot be
    meaningfully solved.
    """

    def solve(
        self, condition_text: str, variables: dict[str, str]
    ) -> ConditionHint | None:
        """Return input hints for both branches, or None if unsolvable."""
        ...
