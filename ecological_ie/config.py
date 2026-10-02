import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-3.8-flash"
STRONG_MODEL = "gemini-3.1-pro-preview"
API_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LST_Gemini")

PRICES_PER_M = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    thinking_level: str = "medium"
    max_output_tokens: int = 65536
    workers: int = 4
    retries: int = 3
    image_resolution: str = "ultra_high"

    @property
    def price_input_per_m(self) -> float:
        return PRICES_PER_M.get(self.model, (2.00, 12.00))[0]

    @property
    def price_output_per_m(self) -> float:
        return PRICES_PER_M.get(self.model, (2.00, 12.00))[1]


def read_env_file(path: Path) -> dict:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def colab_secret(names) -> str:
    try:
        from google.colab import userdata
    except ImportError:
        return ""
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            continue
        if value:
            return value
    return ""


def find_api_key(env_file: Path | None = None) -> str:
    for name in API_KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    env_values = read_env_file(env_file or Path(".env"))
    for name in API_KEY_NAMES:
        if env_values.get(name):
            return env_values[name]
    return colab_secret(API_KEY_NAMES)
