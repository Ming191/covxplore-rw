from __future__ import annotations

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

from covxplore.driver.contract import ContractViolation


_CPP_LANGUAGE = Language(tree_sitter_cpp.language())


class DriverContractValidator:
    """Validate test body before sending it to a concrete executor."""

    def __init__(self) -> None:
        self._parser = Parser(_CPP_LANGUAGE)

    def validate(self, test_body: str) -> list[ContractViolation]:
        violations: list[ContractViolation] = []
        body = test_body.strip()

        if not body:
            violations.append(
                ContractViolation(
                    code="EMPTY_BODY",
                    message="Test body is empty.",
                    severity="ERROR",
                )
            )
            return violations

        if "```" in test_body or "~~~" in test_body:
            violations.append(
                ContractViolation(
                    code="MARKDOWN_FENCE",
                    message="Test body must not contain Markdown code fences.",
                    severity="ERROR",
                )
            )

        source = test_body.encode("utf-8")
        tree = self._parser.parse(source)

        if _has_main_definition(tree.root_node, source):
            violations.append(
                ContractViolation(
                    code="HAS_MAIN",
                    message="Test body must not define main(); AkaUT wraps the body.",
                    severity="ERROR",
                )
            )

        if _has_value_return(tree.root_node, source):
            violations.append(
                ContractViolation(
                    code="RETURN_VALUE",
                    message="Test body is inserted into a void AkaUT harness function; use return; or discard the result.",
                    severity="ERROR",
                )
            )

        if _has_node_type(tree.root_node, "preproc_include"):
            violations.append(
                ContractViolation(
                    code="INCLUDE_DIRECTIVE",
                    message="Test body should not include #include directives; send only the driver body.",
                    severity="WARNING",
                )
            )

        if _has_node_type(tree.root_node, "preproc_ifdef"):
            violations.append(
                ContractViolation(
                    code="INCLUDE_GUARD",
                    message="Test body should not include header/include guards.",
                    severity="WARNING",
                )
            )

        if _has_pragma_once(tree.root_node, source):
            violations.append(
                ContractViolation(
                    code="PRAGMA_ONCE",
                    message="Test body should not include #pragma once header guards.",
                    severity="WARNING",
                )
            )

        marker_index = test_body.find("AKA_ACTUAL_OUTPUT")
        if marker_index == -1:
            violations.append(
                ContractViolation(
                    code="MISSING_ACTUAL_OUTPUT_MARKER",
                    message=(
                        "Missing AKA_ACTUAL_OUTPUT marker; AkaUT may still execute, "
                        "but pre-calling coverage mark insertion can be incomplete."
                    ),
                    severity="WARNING",
                )
            )
        elif ";" not in test_body[:marker_index]:
            violations.append(
                ContractViolation(
                    code="MISSING_SEMICOLON_BEFORE_MARKER",
                    message=(
                        "AKA_ACTUAL_OUTPUT marker requires a semicolon before it so "
                        "AkaUT can insert the pre-calling mark."
                    ),
                    severity="WARNING",
                )
            )

        return violations


def _has_node_type(node: Node, node_type: str) -> bool:
    if node.type == node_type:
        return True
    return any(_has_node_type(child, node_type) for child in node.children)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _has_main_definition(node: Node, source: bytes) -> bool:
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        if declarator is not None and _function_declarator_name(declarator, source) == "main":
            return True
    return any(_has_main_definition(child, source) for child in node.children)


def _has_value_return(node: Node, source: bytes) -> bool:
    if node.type == "return_statement":
        return _node_text(node, source).strip() != "return;"
    return any(_has_value_return(child, source) for child in node.children)


def _function_declarator_name(node: Node, source: bytes) -> str | None:
    if node.type == "identifier":
        return _node_text(node, source)
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        found = _function_declarator_name(declarator, source)
        if found is not None:
            return found
    for child in node.children:
        found = _function_declarator_name(child, source)
        if found is not None:
            return found
    return None


def _has_pragma_once(node: Node, source: bytes) -> bool:
    if node.type == "preproc_call":
        return _node_text(node, source).strip() == "#pragma once"
    return any(_has_pragma_once(child, source) for child in node.children)
