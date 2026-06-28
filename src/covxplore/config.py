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

    # Generation loop
    max_iterations: int = 15
    max_tokens: int = 4000
    llm_temperature: float = 0.4

    llm_empty_retries: int = 5  # extra attempts when a call returns empty
    llm_retry_backoff_sec: float = 1.0  # base backoff, grows linearly per attempt
    mcdc_target: float = 1.0  # 1.0 = 100 % MC/DC coverage
    min_suite_size: int = 1  # keep at least this many tests even if redundant
    redundant_streak_limit: int = 3  # early-stop after N consecutive redundant tests
    fail_streak_limit: int = 5  # early-stop after N consecutive failing tests (0 passing)
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
