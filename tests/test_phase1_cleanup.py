import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covxplore.config import Settings
from covxplore.tools.execute_testcase import FatalToolError


def test_fatal_tool_error_is_runtime_error() -> None:
    assert issubclass(FatalToolError, RuntimeError)


def test_generation_settings_reject_missing_api_key() -> None:
    settings = Settings(deepseek_api_key="")

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        settings.validate_for_generation()


def test_generation_settings_accept_valid_defaults_with_api_key() -> None:
    settings = Settings(deepseek_api_key="test-key")

    settings.validate_for_generation()
