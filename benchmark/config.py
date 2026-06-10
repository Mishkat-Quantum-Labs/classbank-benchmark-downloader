"""Benchmark configuration — loads from .env and defines engine settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── API Keys ────────────────────────────────────────────────────────────────

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# ── AWS Bedrock (speaker classification) ────────────────────────────────────

AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", "us-east-1")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# ── Gemini Engine Settings ──────────────────────────────────────────────────

GEMINI_MODEL_MAP: dict[str, str] = {
    "gemini_pro": os.getenv("GEMINI_STT_MODEL", "gemini-2.5-pro"),
    "gemini_31_pro": os.getenv("GEMINI_31_PRO_MODEL", "gemini-3.1-pro-preview"),
    "gemini_35_flash": os.getenv("GEMINI_35_FLASH_MODEL", "gemini-3.5-flash"),
}

GEMINI_STT_TEMPERATURE: float = float(os.getenv("GEMINI_STT_TEMPERATURE", "0.0"))
GEMINI_STT_TOP_P: float = float(os.getenv("GEMINI_STT_TOP_P", "0.0"))
GEMINI_STT_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_STT_MAX_OUTPUT_TOKENS", "65536"))
GEMINI_STT_POLL_TIMEOUT: int = int(os.getenv("GEMINI_STT_POLL_TIMEOUT", "600"))
GEMINI_STT_POLL_INTERVAL: int = int(os.getenv("GEMINI_STT_POLL_INTERVAL", "10"))
GEMINI_STT_HTTP_TIMEOUT: int = int(os.getenv("GEMINI_STT_HTTP_TIMEOUT", "600000"))
GEMINI_STT_UPLOAD_RETRIES: int = int(os.getenv("GEMINI_STT_UPLOAD_RETRIES", "2"))
GEMINI_STT_UPLOAD_RETRY_DELAY: int = int(os.getenv("GEMINI_STT_UPLOAD_RETRY_DELAY", "15"))

# ── ElevenLabs Engine Settings ──────────────────────────────────────────────

SEGMENT_PAUSE_THRESHOLD_MS: int = int(os.getenv("SEGMENT_PAUSE_THRESHOLD_MS", "2000"))
SEGMENT_MAX_DURATION_SEC: float = float(os.getenv("SEGMENT_MAX_DURATION_SEC", "30.0"))
ELEVENLABS_ROLE_CLASSIFICATION_RETRIES: int = int(
    os.getenv("ELEVENLABS_ROLE_CLASSIFICATION_RETRIES", "3")
)

# ── Engine Registry ─────────────────────────────────────────────────────────

ENGINES: dict[str, str] = {
    "gemini_pro": "Gemini 2.5 Pro",
    "gemini_31_pro": "Gemini 3.1 Pro",
    "gemini_35_flash": "Gemini 3.5 Flash",
    "elevenlabs_scribe_v2": "ElevenLabs Scribe v2",
}

DEFAULT_STT_ENGINE: str = os.getenv("DEFAULT_STT_ENGINE", "gemini_pro")

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
MEDIA_DIR: Path = DATASET_DIR / "media"
PARSED_DIR: Path = DATASET_DIR / "parsed"
RESULTS_DIR: Path = PROJECT_ROOT / "benchmark_results"
