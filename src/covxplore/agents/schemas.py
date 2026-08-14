from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateTestAction(BaseModel):
    """One generated C++ test driver body."""

    model_config = ConfigDict(extra="forbid")

    test_body: str = Field(
        ...,
        description=(
            "Complete C++ test driver body inserted into AkaUT's harness. "
            "Do not include headers, main(), assertions, namespace declarations, "
            "or type definitions."
        ),
    )
    test_name: str | None = Field(
        default=None,
        description="Optional test case name; AkaUT generates one when omitted.",
    )

    @field_validator("test_body")
    @classmethod
    def _body_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("test_name")
    @classmethod
    def _name_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank when provided")
        return value


class GenerateTestBatchAction(BaseModel):
    """A bounded batch produced by every reasoning treatment."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[GenerateTestAction] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "One to eight distinct test candidates. Use fewer when fewer useful "
            "coverage gaps remain."
        ),
    )
