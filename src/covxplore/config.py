"""Runtime settings for covxplore, loaded from environment variables or .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AkaUT REST API
    akaut_base_url: str = "http://localhost:8080"

    # LLM provider selection. "deepseek" remains the default for backward compatibility.
    llm_provider: str = "deepseek"

    # DeepSeek via LiteLLM (backward compatible DEEPSEEK_* settings)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek/deepseek-v4-pro"

    # Kimchi OpenAI-compatible endpoint
    kimchi_api_key: str = ""
    kimchi_base_url: str = "https://llm.kimchi.dev/openai/v1"
    kimchi_model: str = "kimi-k2.6"

    # Langfuse/OpenLIT observability
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Generation loop
    max_batches: int = 15
    # DeepSeek V4 defaults to reasoning_effort="high" (thinking mode always on unless
    # explicitly disabled), and the model's internal "thinking" tokens are billed against
    # this same max_tokens budget as the final content/tool_calls. A low value here risks
    # the response getting cut mid-thinking (finish_reason="length", content/tool_calls
    # empty) before the model ever emits its actual answer — this hits doubly hard for
    # CrewAI's own agent-level `reasoning` feature, since each reasoning attempt is itself
    # an LLM call subject to the same truncation risk on top of the normal task calls.
    max_tokens: int = 8000
    llm_temperature: float = 0.4

    # When enabled, the test_generator agent reflects and drafts a plan before executing
    # each task (CrewAI's built-in "reasoning" feature). This is the toggle that surfaces
    # the agent's chain-of-thought as "Reasoning Started/Completed" panels in the console
    # (requires verbose=True, already the case for the crew/agent below).
    agent_reasoning: bool = False
    # Max reasoning refinement attempts before proceeding regardless of readiness.
    # None means "keep refining until the agent reports ready".
    agent_max_reasoning_attempts: int | None = None

    min_suite_size: int = 1  # keep at least this many tests even if redundant
    redundant_streak_limit: int = 3  # early-stop after N consecutive redundant tests
    fail_streak_limit: int = 3  # early-stop after N consecutive failing batches
    request_timeout_sec: int = 120  # per REST call

    # Ablation
    default_prompt_variant: str = "full"
    ablation_repeat: int = 3  # runs per variant per function


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
