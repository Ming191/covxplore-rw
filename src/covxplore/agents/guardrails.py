from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field, ValidationError

from covxplore.agents.schemas import GenerateTestAction


class ActionValidationResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_tool_input(payload: dict) -> ActionValidationResult:
    """Validate an execute_testcase tool-input payload without executing it."""

    try:
        action = GenerateTestAction.model_validate(payload)
    except ValidationError as exc:
        return ActionValidationResult(
            ok=False,
            errors=[_format_validation_error(error) for error in exc.errors()],
        )

    warnings: list[str] = []
    if action.target_polarity is not None and not action.target_reason:
        warnings.append("target_polarity provided without target_reason")

    return ActionValidationResult(ok=True, warnings=warnings)


def _format_validation_error(error: Mapping[str, Any]) -> str:
    loc = ".".join(str(part) for part in error.get("loc", ())) or "payload"
    return f"{loc}: {error.get('msg', 'invalid value')}"
