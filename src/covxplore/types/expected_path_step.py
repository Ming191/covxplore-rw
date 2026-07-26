from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExpectedPathStep(BaseModel):
    """One step in a predicted branch-outcome trace from function entry to a target decision.

    A full ``expected_path`` is an ordered list of these, one per condition the test body is
    expected to force on its way to the targeted branch (the last step *is* the target itself).
    """

    node_id: int = Field(..., gt=0, description="Branch condition node id evaluated at this step.")
    polarity: Literal["TRUE", "FALSE"] = Field(
        ..., description="Predicted runtime outcome of this condition."
    )
    reason: str | None = Field(
        default=None, description="Optional short reason why this step should take this polarity."
    )
