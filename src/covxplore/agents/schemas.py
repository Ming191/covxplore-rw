from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from covxplore.types.expected_path_step import ExpectedPathStep


class GenerateTestAction(BaseModel):
    """Schema for one generated test execution action."""

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
        description="Optional branch condition node id this test is targeting.",
    )
    target_polarity: Literal["TRUE", "FALSE"] | None = Field(
        default=None,
        description="Optional target branch polarity for target_node_id.",
    )
    target_reason: str | None = Field(
        default=None,
        description="Optional short reason explaining why this target was selected.",
    )
    expected_path: list[ExpectedPathStep] = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED ordered vector of predicted branch outcomes from function entry up to "
            "and including the target decision, e.g. "
            "[{\"node_id\": 12, \"polarity\": \"TRUE\"}, {\"node_id\": 19, \"polarity\": \"FALSE\"}]. "
            "Do NOT leave this empty. The last step is the coverage target "
            "(target_node_id/target_polarity are auto-filled from it when omitted). "
            "Only include decisions this test body actually forces. After execution, the first "
            "step whose predicted polarity was NOT observed at runtime is reported back as a "
            "path divergence — use that to see exactly where your prediction was wrong."
        ),
    )

    @field_validator("test_body")
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

        if self.expected_path:
            last = self.expected_path[-1]
            if self.target_node_id is None:
                # Let expected_path double as the sole source of target metadata so
                # every existing target_node_id/target_polarity consumer keeps working.
                self.target_node_id = last.node_id
                self.target_polarity = last.polarity
                if self.target_reason is None:
                    self.target_reason = last.reason
            elif self.target_node_id != last.node_id or self.target_polarity != last.polarity:
                # Agents occasionally emit a mismatched last step; prefer the explicit
                # target_* fields and rewrite the path tail so the tool call is not rejected.
                self.expected_path[-1] = ExpectedPathStep(
                    node_id=self.target_node_id,
                    polarity=self.target_polarity,
                    reason=last.reason or self.target_reason,
                )
        return self


class GenerateTestBatchAction(BaseModel):
    """Schema for one batch of generated test execution actions."""

    candidates: list[GenerateTestAction] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "Batch of 3-5 distinct test candidates to execute concurrently; use fewer only when "
            "fewer useful targets remain. Every candidate MUST include a non-empty expected_path."
        ),
    )


class GenerateTestBatchSingleAction(BaseModel):
    """Ablation: at most one candidate per batch (no batch diversity)."""

    candidates: list[GenerateTestAction] = Field(
        ...,
        min_length=1,
        max_length=1,
        description=(
            "Exactly one test candidate for this session. Every candidate MUST include a "
            "non-empty expected_path."
        ),
    )


class GenerateTestBatchAnyAction(BaseModel):
    """Schema for exploratory one-shot batches without a fixed candidate cap."""

    candidates: list[GenerateTestAction] = Field(
        ...,
        min_length=1,
        description="Batch of distinct test candidates; cover distinct statement and branch gaps and avoid duplicates.",
    )


class GenerateTestActionOptionalPath(BaseModel):
    """Same as GenerateTestAction but expected_path is optional (no-path ablation)."""

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
        description="Optional branch condition node id this test is targeting.",
    )
    target_polarity: Literal["TRUE", "FALSE"] | None = Field(
        default=None,
        description="Optional target branch polarity for target_node_id.",
    )
    target_reason: str | None = Field(
        default=None,
        description="Optional short reason explaining why this target was selected.",
    )
    expected_path: list[ExpectedPathStep] = Field(
        default_factory=list,
        description=(
            "Optional ordered vector of predicted branch outcomes. Leave empty in no-path "
            "ablation runs. When provided, the last step is the coverage target."
        ),
    )

    @field_validator("test_body")
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

        if self.expected_path:
            last = self.expected_path[-1]
            if self.target_node_id is None:
                self.target_node_id = last.node_id
                self.target_polarity = last.polarity
                if self.target_reason is None:
                    self.target_reason = last.reason
            elif self.target_node_id != last.node_id or self.target_polarity != last.polarity:
                self.expected_path[-1] = ExpectedPathStep(
                    node_id=self.target_node_id,
                    polarity=self.target_polarity,
                    reason=last.reason or self.target_reason,
                )
        return self


class GenerateTestBatchOptionalPathAction(BaseModel):
    """Batch schema for no-path ablation (expected_path optional per candidate)."""

    candidates: list[GenerateTestActionOptionalPath] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "Batch of 3-5 distinct test candidates to execute concurrently; use fewer only when "
            "fewer useful targets remain. expected_path is optional."
        ),
    )
