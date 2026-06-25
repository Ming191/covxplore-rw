"""Z3GuidanceSolver — reusable building-blocks for agent-prompt injection.

Takes a single condition text + CDT variable map and returns human-readable
hints for the TRUE/FALSE branches.  Does NOT generate test bodies — the LLM
handles struct/context construction.
"""

from __future__ import annotations

from covxplore.concolic import (
    _RE_COMPARE,
    _RE_BOOL_VAR,
    _resolve_var_type,
    _sanitise,
    _to_z3_type,
    solve_pair,
    ParsedCondition,
)
from covxplore.concolic.protocol import ConditionHint


class Z3GuidanceSolver:
    """Solves single conditions for prompt-injectable constraint hints.

    Example
    -------
    >>> solver = Z3GuidanceSolver()
    >>> hint = solver.solve("p->indexNext > 1", {"p->indexNext": "int"})
    >>> hint.true_branch
    "p->indexNext >= 2"
    >>> hint.false_branch
    "p->indexNext <= 1"
    """

    # ------------------------------------------------------------------
    # Parsing (same regex approach as parse_conditions, single-input)
    # ------------------------------------------------------------------

    def _parse_one(self, text: str, variables: dict[str, str]) -> ParsedCondition | None:
        """Parse a *single* condition string into a ``ParsedCondition``."""
        text = text.strip()

        # Comparison: "var op const"
        m = _RE_COMPARE.search(text)
        if m:
            var_raw, op, rhs_str = m.group(1), m.group(2), m.group(3)
            var_type = _resolve_var_type(var_raw, variables)
            z3_type = _to_z3_type(var_type)
            if z3_type is None:
                return None
            return ParsedCondition(
                node_id=None,
                raw=text,
                var_name=_sanitise(var_raw),
                member_name=var_raw.split("->")[-1].split(".")[-1],
                op=op,
                rhs=int(rhs_str),
                display=f"{var_raw} {op} {rhs_str}",
                z3_type=z3_type,
            )

        # Pure boolean: "var" or "!var"
        m = _RE_BOOL_VAR.match(text)
        if m:
            var_raw = m.group(1)
            negated = text.startswith("!")
            var_type = _resolve_var_type(var_raw, variables)
            if "bool" not in var_type.lower():
                return None
            return ParsedCondition(
                node_id=None,
                raw=text,
                var_name=_sanitise(var_raw),
                member_name=var_raw.split("->")[-1].split(".")[-1],
                op="!bool" if negated else "bool",
                rhs=0,
                display=f"{var_raw} == {'true' if not negated else 'false'}",
                z3_type="Bool",
            )

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, condition_text: str, variables: dict[str, str]
    ) -> ConditionHint | None:
        """Return constraint hints for both branches, or ``None`` if unsolvable."""
        parsed = self._parse_one(condition_text, variables)
        if parsed is None:
            return None

        true_val, false_val = solve_pair(parsed)

        if true_val is None and false_val is None:
            return ConditionHint(
                node_id=None,
                is_solvable=False,
                true_branch="",
                false_branch="",
            )

        return ConditionHint(
            node_id=None,
            is_solvable=True,
            true_branch=self._fmt_hint(parsed, True, true_val),
            false_branch=self._fmt_hint(parsed, False, false_val),
        )

    # ------------------------------------------------------------------
    # Hint formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_hint(
        cond: ParsedCondition,
        polarity: bool,
        value: int | bool | None,
    ) -> str:
        """Format a single branch hint like ``p->indexNext=2``."""
        if value is None:
            return "?"

        member = cond.member_name

        # For struct fields: p->indexNext=2
        if "->" in cond.display or "." in cond.display:
            prefix = cond.display.split()[0]  # p->indexNext
        else:
            prefix = member

        if cond.z3_type == "Bool":
            cpp_val = "true" if value else "false"
        else:
            cpp_val = str(value)

        return f"{prefix}={cpp_val}"
