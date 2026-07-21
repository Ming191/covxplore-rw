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
    deepseek_model: str = "deepseek/deepseek-chat"

    # Kimchi OpenAI-compatible endpoint
    kimchi_api_key: str = ""
    kimchi_base_url: str = "https://llm.kimchi.dev/openai/v1"
    kimchi_model: str = "kimi-k2.6"

    # Legacy MAS-enhanced runner compatibility. COV127 entry points do not read
    # these coverage/iteration controls.
    max_iterations: int = 15
    # No Covxplore-imposed output cap by default. Provider/model limits still apply.
    max_tokens: int | None = None
    llm_temperature: float = 0.4

    llm_empty_retries: int = 5  # extra attempts when a call returns empty
    llm_retry_backoff_sec: float = 1.0  # base backoff, grows linearly per attempt
    mcdc_target: float = 1.0  # 1.0 = 100 % MC/DC coverage
    min_suite_size: int = 1  # keep at least this many tests even if redundant
    redundant_streak_limit: int = 3  # early-stop after N consecutive redundant tests
    fail_streak_limit: int = 5  # early-stop after N consecutive failing tests (0 passing)
    agent_retry_limit: int = 2  # retry entire agent loop if it returns 0 tests
    request_timeout_sec: int = 120  # per REST call

    # COV127 neuro-symbolic generation
    strategy: str = "hybrid"
    statement_target: float = 1.0
    branch_target: float = 1.0
    scheduler_mode: str = "rule"
    wall_time_minutes: float = 30.0
    max_llm_calls: int = 15
    max_symbolic_attempts: int = 30
    max_test_executions: int = 30

    # Legacy prompt registry plus COV127 experiment repetition default.
    default_prompt_variant: str = "full"
    ablation_repeat: int = 3  # runs per variant per function


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
