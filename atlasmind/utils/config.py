import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    GITHUB_TOKEN: str
    SERPAPI_KEY: str
    DEEPGRAM_API_KEY: str
    GEMINI_API_KEY: str
    
    WIKI_API: str = "https://en.wikipedia.org/w/api.php"

    def __post_init__(self):
        if not self.GITHUB_TOKEN:
            raise ValueError("Missing GITHUB_TOKEN. Please set it in your .env file.")
        if not self.SERPAPI_KEY:
            raise ValueError("Missing WIKI_API. Please set it in your .env file.")
        if not self.DEEPGRAM_API_KEY:
            raise ValueError("Missing DEEPGRAM_API_KEY. Please set it in your .env file.")
        if not self.GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY. Please set it in your .env file.")

# Build safely: if env var missing, raise right here
def _get_env_str(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return str(value)


settings = Settings(
    GITHUB_TOKEN=_get_env_str("GITHUB_TOKEN"),
    SERPAPI_KEY=_get_env_str('SERPAPI_API_KEY'),
    DEEPGRAM_API_KEY=_get_env_str("DEEPGRAM_API_KEY"),
    GEMINI_API_KEY=_get_env_str("GEMINI_API_KEY"),
)

@dataclass(frozen=True)
class CodeExecutorSettings:
    """Settings for code fetching (Hugging Face) and execution (RapidAPI)."""


    # RapidAPI / Judge0 execution
    RAPIDAPI_URL: str = "https://judge0-ce.p.rapidapi.com/submissions"
    RAPIDAPI_HOST: str = "judge0-ce.p.rapidapi.com"
    RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
    
    def __post_init__(self):
        if not self.RAPIDAPI_KEY:
            raise ValueError("Missing RAPIDAPI_KEY. Please set it in your .env file.")


# Singleton-style instance
code_execution_settings = CodeExecutorSettings()
