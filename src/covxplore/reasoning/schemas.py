from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from covxplore.agents.schemas import GenerateTestAction, GenerateTestBatchAction


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _decode_test_body(value: str) -> str:
    return (
        value.replace("__RAW_BEGIN__", 'R"AKA(')
        .replace("__RAW_END__", ')AKA"')
        .replace("R__AKA__(", 'R"AKA(')
        .replace(")__AKA__", ')AKA"')
        .replace(")AKA__", ')AKA"')
        .replace("__DQ__", '"')
        .replace("__BS__", "\\")
        .replace("__NL__", "\n")
    )


class EncodedTestAction(_StrictModel):
    test_body_encoded: str = Field(..., min_length=1)
    test_name: str | None = None

    def decode(self) -> GenerateTestAction:
        return GenerateTestAction(
            test_name=self.test_name,
            test_body=_decode_test_body(self.test_body_encoded),
        )


class EncodedTestBatchResult(_StrictModel):
    candidates: list[EncodedTestAction] = Field(..., min_length=1, max_length=8)

    def decode(self) -> GenerateTestBatchAction:
        return GenerateTestBatchAction(candidates=[item.decode() for item in self.candidates])


class EncodedChainOfThoughtResult(_StrictModel):
    reasoning: str = Field(..., min_length=1)
    candidates: list[EncodedTestAction] = Field(..., min_length=1, max_length=8)


class ChainOfThoughtResult(_StrictModel):
    reasoning: str = Field(..., min_length=1)
    candidates: list[GenerateTestAction] = Field(..., min_length=1, max_length=8)


class PathGuidedCandidate(GenerateTestAction):
    path_id: str = Field(..., min_length=1)


class EncodedPathGuidedCandidate(EncodedTestAction):
    path_id: str = Field(..., min_length=1)

    def decode(self) -> PathGuidedCandidate:
        return PathGuidedCandidate(
            test_name=self.test_name,
            test_body=_decode_test_body(self.test_body_encoded),
            path_id=self.path_id,
        )


class EncodedPathGuidedResult(_StrictModel):
    candidates: list[EncodedPathGuidedCandidate] = Field(..., min_length=1, max_length=5)

    def decode(self) -> "PathGuidedResult":
        return PathGuidedResult(candidates=[item.decode() for item in self.candidates])


class PathGuidedResult(_StrictModel):
    candidates: list[PathGuidedCandidate] = Field(..., min_length=1, max_length=5)


class ContextAction(_StrictModel):
    action: Literal["search_nodes", "get_node_source", "done"]
    query: str | None = None
    types: list[str] = Field(default_factory=list, max_length=8)
    absolute_path: str | None = None
