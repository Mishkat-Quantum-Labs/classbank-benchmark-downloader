"""LLM client — unified model factory using LangChain init_chat_model.

Single entry point for all LLM models across the benchmark pipeline.
Uses init_chat_model with provider-specific packages:
  - langchain-aws (ChatBedrockConverse) for AWS Bedrock models
  - langchain-google-genai (ChatGoogleGenerativeAI) for Google Gemini models
"""

import logging
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from json_repair import repair_json
from langchain.chat_models import init_chat_model

from benchmark.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BEDROCK_REGION,
    ELEVENLABS_ROLE_CLASSIFICATION_RETRIES,
    GEMINI_API_KEY,
    GEMINI_MODEL_MAP,
    GEMINI_STT_MAX_OUTPUT_TOKENS,
    GEMINI_STT_TEMPERATURE,
    GEMINI_STT_TOP_P,
)

logger = logging.getLogger("benchmark")

# ── Bedrock client config ───────────────────────────────────────────────────

BEDROCK_CONFIG = BotoConfig(
    read_timeout=300,
    connect_timeout=10,
    retries={"max_attempts": 5, "mode": "adaptive"},
)


def _get_bedrock_client():
    """Create a Bedrock runtime client with configured credentials."""
    return boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=BEDROCK_CONFIG,
    )


# ── Model registry ─────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # AWS Bedrock models
    "claude_haiku": {
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "model_provider": "bedrock_converse",
        "temperature": 0.0,
        "max_tokens": 256,
    },
    "claude_sonnet": {
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "model_provider": "bedrock_converse",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    # Google Gemini models
    "gemini_pro": {
        "model": GEMINI_MODEL_MAP.get("gemini_pro", "gemini-2.5-pro"),
        "model_provider": "google_genai",
        "temperature": GEMINI_STT_TEMPERATURE,
        "max_output_tokens": GEMINI_STT_MAX_OUTPUT_TOKENS,
    },
    "gemini_31_pro": {
        "model": GEMINI_MODEL_MAP.get("gemini_31_pro", "gemini-3.1-pro-preview"),
        "model_provider": "google_genai",
        "temperature": GEMINI_STT_TEMPERATURE,
        "max_output_tokens": GEMINI_STT_MAX_OUTPUT_TOKENS,
    },
    "gemini_35_flash": {
        "model": GEMINI_MODEL_MAP.get("gemini_35_flash", "gemini-3.5-flash"),
        "model_provider": "google_genai",
        "temperature": GEMINI_STT_TEMPERATURE,
        "max_output_tokens": GEMINI_STT_MAX_OUTPUT_TOKENS,
    },
}


# ── Unified model factory ──────────────────────────────────────────────────


def get_chat_model(model_key: str = "claude_haiku"):
    """Initialize and return a LangChain chat model via init_chat_model.

    This is the single entry point for all LLM usage in the benchmark.
    Routes to the correct provider (bedrock_converse or google_genai)
    based on the model registry.

    Args:
        model_key: Key into MODEL_REGISTRY. Options:
            - "claude_haiku" (Bedrock, speaker classification)
            - "claude_sonnet" (Bedrock, semantic WER evaluation)
            - "gemini_pro" (Google, transcription)
            - "gemini_31_pro" (Google, transcription)
            - "gemini_35_flash" (Google, transcription)

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If model_key is not in the registry.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_key]
    provider = config["model_provider"]

    if provider == "bedrock_converse":
        return init_chat_model(
            model=config["model"],
            model_provider="bedrock_converse",
            region_name=BEDROCK_REGION,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            client=_get_bedrock_client(),
        )

    elif provider == "google_genai":
        return init_chat_model(
            model=config["model"],
            model_provider="google_genai",
            api_key=GEMINI_API_KEY,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            top_p=GEMINI_STT_TOP_P,
        )

    else:
        raise ValueError(f"Unsupported provider '{provider}' for model '{model_key}'")


# ── Speaker classification (ElevenLabs pipeline) ───────────────────────────


def classify_speakers(transcript: str, speaker_count: int) -> dict[str, str]:
    """Use Haiku to classify speakers as Teacher/Observer/Student.

    Args:
        transcript: Raw transcript with "Speaker N" labels.
        speaker_count: Number of unique speakers.

    Returns:
        Dict mapping "Speaker N" → role label (e.g. "Teacher", "Student 1").

    Raises:
        RuntimeError: If classification fails after max retries.
    """
    from benchmark.prompts import build_speaker_classification_prompt

    prompt = build_speaker_classification_prompt(transcript, speaker_count)
    llm = get_chat_model("claude_haiku")

    max_attempts = ELEVENLABS_ROLE_CLASSIFICATION_RETRIES
    last_raw = ""

    for attempt in range(max_attempts):
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        last_raw = raw

        parsed = repair_json(raw, return_objects=True)
        if isinstance(parsed, dict):
            return parsed

        logger.warning(
            "[LLM] Speaker classification returned invalid JSON (attempt %d/%d): %s",
            attempt + 1,
            max_attempts,
            raw[:200],
        )

    raise RuntimeError(
        f"Speaker role classification failed after {max_attempts} attempts. "
        f"Last response: {last_raw[:500]}"
    )
