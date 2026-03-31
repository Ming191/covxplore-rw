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

    # DeepSeek via LiteLLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek/deepseek-reasoner"

    # Generation loop
    max_iterations: int = 15
    max_tokens: int = 4000
    mcdc_target: float = 1.0  # 1.0 = 100 % MC/DC coverage
    min_suite_size: int = 1  # keep at least this many tests even if redundant
    redundant_streak_limit: int = 3  # early-stop after N consecutive redundant tests
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
