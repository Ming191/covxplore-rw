from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from covxplore.types.path_prediction import ExpectedPath, ExpectedPathStep, format_compact_expected_path


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


class PathGuidedTestAction(GenerateTestAction):
    path_id: str = Field(..., min_length=1)
    expected_path: ExpectedPath

    @field_serializer("expected_path")
    def _serialize_expected_path(self, steps: list[ExpectedPathStep]) -> str:
        return format_compact_expected_path(steps)


class PathGuidedTestBatchAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[PathGuidedTestAction] = Field(..., min_length=1, max_length=5)


class GenerateTestBatchAction(BaseModel):
    """A bounded batch produced by every reasoning treatment."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[PathGuidedTestAction | GenerateTestAction] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "One to eight distinct test candidates. Use fewer when fewer useful "
            "coverage gaps remain. Path-guided batches include path_id and expected_path."
        ),
    )
