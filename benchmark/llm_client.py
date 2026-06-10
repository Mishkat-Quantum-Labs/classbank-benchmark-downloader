"""LLM client — initializes chat models for speaker classification.

Uses AWS Bedrock with Claude Haiku for cost-effective speaker role classification
in the ElevenLabs pipeline.
"""

import logging

import boto3
from botocore.config import Config as BotoConfig
from json_repair import repair_json
from langchain.chat_models import init_chat_model

from benchmark.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BEDROCK_REGION,
    ELEVENLABS_ROLE_CLASSIFICATION_RETRIES,
)

logger = logging.getLogger("benchmark")

# ── Bedrock client config ───────────────────────────────────────────────────

BEDROCK_CONFIG = BotoConfig(
    read_timeout=300,
    connect_timeout=10,
    retries={"max_attempts": 5, "mode": "adaptive"},
)

# ── Model definitions ───────────────────────────────────────────────────────

TEXT_MODELS = {
    "claude_haiku": {
        "name": "Claude Haiku 4.5",
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "provider": "anthropic",
        "max_tokens": 256,
        "temperature": 0.0,
    },
    "claude_sonnet": {
        "name": "Claude Sonnet 4.5",
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
}


def _get_bedrock_client():
    """Create a Bedrock runtime client with configured credentials."""
    return boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=BEDROCK_CONFIG,
    )


def get_chat_model(model_key: str = "claude_haiku"):
    """Initialize and return a LangChain chat model for the given model key.

    Args:
        model_key: Key into TEXT_MODELS dict. Default "claude_haiku".

    Returns:
        A LangChain chat model instance configured for Bedrock.
    """
    model_config = TEXT_MODELS[model_key]
    return init_chat_model(
        model_config["model_id"],
        model_provider="bedrock",
        region_name=BEDROCK_REGION,
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        client=_get_bedrock_client(),
    )


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
