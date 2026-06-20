from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Severity = Literal["WARNING", "ERROR"]


@dataclass(frozen=True)
class ContractViolation:
    """Machine-readable violation of the Python → driver body contract."""

    code: str
    message: str
    severity: Severity
