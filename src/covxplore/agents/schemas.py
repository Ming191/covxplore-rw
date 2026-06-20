from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self


class GenerateTestAction(BaseModel):
    """Schema for one generated test execution action."""

    run_id: str = Field(
        ...,
        description=(
            "Opaque generation run id provided in the task instructions. "
            "Pass it unchanged on every execute_testcase call."
        ),
    )
    absolute_path: str = Field(
        ...,
        description=(
            "Absolute path of the function node being tested "
            "(same value used throughout a generation run)."
        ),
    )
    test_body: str = Field(
        ...,
        description=(
            "Complete C++ test driver body to be placed inside AkaUT's test "
            "harness. Must be valid C++ that calls the function under test and "
            "sets up all required input variables. Do NOT include main() or "
            "include guards — AkaUT wraps the body automatically."
        ),
    )
    test_name: str | None = Field(
        default=None,
        description="Optional test case name (auto-generated if omitted).",
    )
    target_node_id: int | None = Field(
        default=None,
        description="Optional MC/DC condition node id this test is targeting.",
    )
    target_polarity: Literal["TRUE", "FALSE"] | None = Field(
        default=None,
        description="Optional target branch polarity for target_node_id.",
    )
    target_reason: str | None = Field(
        default=None,
        description="Optional short reason explaining why this target was selected.",
    )

    @field_validator("run_id", "absolute_path", "test_body")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("test_name", "target_reason")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank when provided")
        return value

    @model_validator(mode="after")
    def _validate_target_metadata(self) -> Self:
        if self.target_polarity is not None and self.target_node_id is None:
            raise ValueError("target_polarity requires target_node_id")
        if self.target_node_id is not None and self.target_polarity is None:
            raise ValueError("target_node_id requires target_polarity")
        if self.target_node_id is not None and self.target_node_id <= 0:
            raise ValueError("target_node_id must be positive")
        return self
