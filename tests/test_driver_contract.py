from covxplore.driver import DriverContractValidator


def _codes(body: str) -> dict[str, str]:
    return {
        violation.code: violation.severity
        for violation in DriverContractValidator().validate(body)
    }


def test_empty_body_rejected():
    codes = _codes("  \n\t")

    assert codes["EMPTY_BODY"] == "ERROR"


def test_markdown_fence_rejected():
    codes = _codes("```cpp\nf();\n```")

    assert codes["MARKDOWN_FENCE"] == "ERROR"


def test_tilde_markdown_fence_rejected():
    codes = _codes("~~~cpp\nf();\n~~~")

    assert codes["MARKDOWN_FENCE"] == "ERROR"


def test_main_function_rejected():
    codes = _codes("int main() { return 0; }")

    assert codes["HAS_MAIN"] == "ERROR"


def test_main_text_in_comment_or_string_is_not_rejected():
    codes = _codes('// int main() { return 0; }\nconst char* s = "void main()";')

    assert "HAS_MAIN" not in codes


def test_missing_actual_output_marker_is_warning():
    codes = _codes("int x = 1; target(x);")

    assert codes["MISSING_ACTUAL_OUTPUT_MARKER"] == "WARNING"


def test_missing_semicolon_before_marker_is_warning():
    codes = _codes("target(x)\nAKA_ACTUAL_OUTPUT = x;")

    assert codes["MISSING_SEMICOLON_BEFORE_MARKER"] == "WARNING"


def test_include_directive_and_guard_are_warnings():
    codes = _codes("#ifndef X\n#define X\n#include <x.h>\n#endif\nint x = 1;")

    assert codes["INCLUDE_DIRECTIVE"] == "WARNING"
    assert codes["INCLUDE_GUARD"] == "WARNING"


def test_include_text_in_comment_or_string_is_not_warning():
    codes = _codes('// #include <x.h>\nconst char* s = "#include <y.h>";')

    assert "INCLUDE_DIRECTIVE" not in codes


def test_pragma_once_is_warning():
    codes = _codes("#pragma once\nint x = 1;")

    assert codes["PRAGMA_ONCE"] == "WARNING"


def test_valid_body_has_no_errors():
    violations = DriverContractValidator().validate(
        "int x = 1;\nint y = target(x);\nAKA_ACTUAL_OUTPUT = y;"
    )

    assert [v for v in violations if v.severity == "ERROR"] == []
