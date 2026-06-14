from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = ""
    types: list[str] | None = Field(default_factory=lambda: ["FUNCTION"])


class RunCreateRequest(BaseModel):
    absolutePath: str
    variant: str = "full"
    maxIterations: int = Field(default=15, ge=1, le=100)
    maxTests: int | None = Field(default=None, ge=1, le=100)
    mcdcTarget: float = Field(default=1.0, ge=0.0, le=1.0)
    outDir: str = "results"
