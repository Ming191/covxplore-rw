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

    def validate_for_generation(self) -> None:
        """Fail fast when required generation settings are missing or invalid."""
        missing: list[str] = []
        if not self.deepseek_api_key.strip():
            missing.append("DEEPSEEK_API_KEY")
        if not self.akaut_base_url.strip():
            missing.append("AKAUT_BASE_URL")
        if not self.deepseek_base_url.strip():
            missing.append("DEEPSEEK_BASE_URL")
        if not self.deepseek_model.strip():
            missing.append("DEEPSEEK_MODEL")

        if missing:
            raise ValueError("Missing required configuration: " + ", ".join(missing))
        if self.max_iterations <= 0:
            raise ValueError("MAX_ITERATIONS must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be > 0")
        if self.request_timeout_sec <= 0:
            raise ValueError("REQUEST_TIMEOUT_SEC must be > 0")
        if not 0.0 <= self.mcdc_target <= 1.0:
            raise ValueError("MCDC_TARGET must be between 0.0 and 1.0")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
