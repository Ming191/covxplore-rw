from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageGap:
    """Rendered coverage guidance plus stable facts used to build it."""
    text: str
    has_mcdc: bool
    mcdc_done: bool
    statement_done: bool
    branch_done: bool
