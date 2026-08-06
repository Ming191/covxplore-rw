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

    # OpenAI-compatible local/custom endpoint
    local_api_key: str = ""
    local_base_url: str = "http://localhost:8000/v1"
    local_model: str = "local-model"

    # Generation loop
    context_version: str = "v1"
    max_batches: int = 15
    # Provider thinking tokens share this budget with the structured response.
    max_tokens: int = 8000
    llm_temperature: float = 0.4
    llm_timeout_sec: int = 180
    llm_max_retries: int = 0
    llm_seed: int | None = None
    llm_thinking: bool | None = None
    prompt_version: str = "reasoning-v1"

    min_suite_size: int = 1  # keep at least this many tests even if redundant
    redundant_streak_limit: int = 3  # early-stop after N consecutive redundant tests
    fail_streak_limit: int = 3  # early-stop after N consecutive failing batches
    request_timeout_sec: int = 120  # per REST call

    # Ablation
    default_prompt_variant: str = "none"
    ablation_repeat: int = 3  # runs per variant per function


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
