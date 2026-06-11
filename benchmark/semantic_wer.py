"""Semantic WER — Multi-turn LLM-as-judge using Claude Sonnet 4.5 via AWS Bedrock.

Exact implementation of Pipecat's open-source stt-benchmark semantic WER approach,
adapted for AWS Bedrock Converse API instead of Anthropic SDK directly.

Source: https://github.com/pipecat-ai/stt-benchmark
License: MIT

Architecture:
1. System prompt (~4000 words) with normalization rules + 8 few-shot examples
2. Claude processes reference + hypothesis in multi-turn conversation
3. Claude follows: NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT → CALCULATE
4. Claude calls calculate_wer tool with structured counts
5. Code does the math: (S + D + I) / N
6. Returns WER + optional reasoning trace for debugging
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

from benchmark.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BEDROCK_REGION,
    SEMANTIC_WER_MAX_RETRIES,
    SEMANTIC_WER_MAX_TOKENS,
    SEMANTIC_WER_MAX_TURNS,
    SEMANTIC_WER_MODEL,
    SEMANTIC_WER_RETRY_BASE_WAIT,
    SEMANTIC_WER_TEMPERATURE,
)
from benchmark.prompts import (
    SEMANTIC_WER_SYSTEM_PROMPT,
    build_semantic_wer_user_prompt,
)

logger = logging.getLogger("benchmark")

# ── Bedrock Client Config ───────────────────────────────────────────────────

_BEDROCK_CONFIG = BotoConfig(
    read_timeout=300,
    connect_timeout=10,
    retries={"max_attempts": 3, "mode": "adaptive"},
)


# ── Tool Definition (exact Pipecat schema) ──────────────────────────────────

CALCULATE_WER_TOOL = {
    "toolSpec": {
        "name": "calculate_wer",
        "description": (
            "Calculate Word Error Rate from error counts. Call this ONCE after you "
            "have normalized, aligned, and verified the texts. "
            "WER = (substitutions + deletions + insertions) / reference_words"
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "substitutions": {
                        "type": "integer",
                        "description": "Number of word substitutions (different words at same position)",
                    },
                    "deletions": {
                        "type": "integer",
                        "description": "Number of word deletions (words in reference missing from hypothesis)",
                    },
                    "insertions": {
                        "type": "integer",
                        "description": "Number of word insertions (extra words in hypothesis not in reference)",
                    },
                    "reference_words": {
                        "type": "integer",
                        "description": "Total word count in normalized reference text",
                    },
                    "normalized_reference": {
                        "type": "string",
                        "description": "The normalized reference text (for verification)",
                    },
                    "normalized_hypothesis": {
                        "type": "string",
                        "description": "The normalized hypothesis text (for verification)",
                    },
                    "errors": {
                        "type": "array",
                        "description": "List of identified errors",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["substitution", "deletion", "insertion"],
                                },
                                "reference": {
                                    "type": "string",
                                    "description": "Reference word (null for insertion)",
                                },
                                "hypothesis": {
                                    "type": "string",
                                    "description": "Hypothesis word (null for deletion)",
                                },
                                "position": {
                                    "type": "integer",
                                    "description": "Position in alignment",
                                },
                            },
                        },
                    },
                },
                "required": ["substitutions", "deletions", "insertions", "reference_words"],
            }
        },
    }
}


# ── Data Classes ────────────────────────────────────────────────────────────


@dataclass
class SemanticWERResult:
    """Result of a semantic WER evaluation."""

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    total_errors: int
    normalized_reference: Optional[str] = None
    normalized_hypothesis: Optional[str] = None
    errors: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)
    num_turns: int = 0
    duration_ms: int = 0


# ── Bedrock Client ──────────────────────────────────────────────────────────


def _get_bedrock_client():
    """Create a Bedrock runtime client for semantic WER evaluation."""
    return boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=_BEDROCK_CONFIG,
    )


# ── Core Evaluation ─────────────────────────────────────────────────────────


def _calculate_wer(
    substitutions: int,
    deletions: int,
    insertions: int,
    reference_words: int,
) -> dict:
    """Programmatic WER calculation — the only non-LLM logic."""
    if reference_words == 0:
        wer = 0.0 if (substitutions + deletions + insertions) == 0 else float("inf")
    else:
        wer = (substitutions + deletions + insertions) / reference_words

    return {
        "wer": wer,
        "wer_percentage": f"{wer:.2%}",
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": reference_words,
        "total_errors": substitutions + deletions + insertions,
    }


def _converse_with_retry(client, request_payload: dict) -> dict:
    """Make a Bedrock Converse API call with retry on transient errors."""
    for attempt in range(1, SEMANTIC_WER_MAX_RETRIES + 1):
        try:
            return client.converse(**request_payload)
        except client.exceptions.ThrottlingException:
            if attempt == SEMANTIC_WER_MAX_RETRIES:
                raise
            wait = SEMANTIC_WER_RETRY_BASE_WAIT * (2 ** (attempt - 1))
            logger.warning(
                "[SemWER] Rate limited, waiting %ds (attempt %d/%d)",
                wait, attempt, SEMANTIC_WER_MAX_RETRIES,
            )
            time.sleep(wait)
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("ThrottlingException", "ServiceUnavailableException"):
                if attempt == SEMANTIC_WER_MAX_RETRIES:
                    raise
                wait = SEMANTIC_WER_RETRY_BASE_WAIT * (2 ** (attempt - 1))
                logger.warning(
                    "[SemWER] %s, waiting %ds (attempt %d/%d)",
                    error_code, wait, attempt, SEMANTIC_WER_MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Unreachable")


def evaluate_semantic_wer(
    reference: str,
    hypothesis: str,
) -> SemanticWERResult:
    """Evaluate semantic WER using Pipecat's multi-turn tool-use approach.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR engine hypothesis transcription.

    Returns:
        SemanticWERResult with WER, error breakdown, and reasoning trace.

    Raises:
        RuntimeError: If evaluation fails after max turns/retries.
    """
    start_time = time.time()

    # Handle empty cases
    if not reference.strip() and not hypothesis.strip():
        return SemanticWERResult(
            wer=0.0, substitutions=0, deletions=0, insertions=0,
            reference_words=0, total_errors=0,
        )
    if not reference.strip():
        words = len(hypothesis.split())
        return SemanticWERResult(
            wer=float("inf"), substitutions=0, deletions=0, insertions=words,
            reference_words=0, total_errors=words,
        )
    if not hypothesis.strip():
        words = len(reference.split())
        return SemanticWERResult(
            wer=1.0, substitutions=0, deletions=words, insertions=0,
            reference_words=words, total_errors=words,
        )

    client = _get_bedrock_client()

    # Build user prompt
    user_prompt = build_semantic_wer_user_prompt(reference, hypothesis)

    # Initialize conversation
    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    reasoning_trace = []
    result = None
    num_turns = 0

    # Base request config — with prompt caching
    # Bedrock processes blocks in order: tools → system → messages.
    # TTLs must be non-increasing in processing order, so tools cachePoint
    # must have TTL >= system cachePoint TTL.
    system_prompt = [
        {"text": SEMANTIC_WER_SYSTEM_PROMPT},
        {"cachePoint": {"type": "default"}},
    ]
    tool_config = {
        "tools": [
            CALCULATE_WER_TOOL,
            {"cachePoint": {"type": "default"}},
        ]
    }

    # Multi-turn conversation loop
    while num_turns < SEMANTIC_WER_MAX_TURNS:
        num_turns += 1

        request_payload = {
            "modelId": SEMANTIC_WER_MODEL,
            "system": system_prompt,
            "messages": messages,
            "toolConfig": tool_config,
            "inferenceConfig": {
                "maxTokens": SEMANTIC_WER_MAX_TOKENS,
                "temperature": SEMANTIC_WER_TEMPERATURE,
            },
        }

        # Disable extended thinking for Claude Sonnet 4.5 — it causes
        # tool_use/tool_result protocol issues with the Converse API
        # and is unnecessary for WER evaluation.
        if "sonnet-4" in SEMANTIC_WER_MODEL or "claude-sonnet" in SEMANTIC_WER_MODEL:
            request_payload["additionalModelRequestFields"] = {
                "thinking": {"type": "disabled"}
            }

        response = _converse_with_retry(client, request_payload)

        # Extract assistant response
        assistant_message = response["output"]["message"]
        stop_reason = response["stopReason"]

        # Store trace
        reasoning_trace.append({
            "role": "assistant",
            "content": assistant_message["content"],
            "stop_reason": stop_reason,
        })

        # Add assistant message to conversation
        messages.append(assistant_message)

        # Check stop reason
        if stop_reason == "end_turn":
            logger.warning("[SemWER] Model finished without calling calculate_wer")
            break

        if stop_reason == "tool_use":
            # Process tool calls — MUST provide a tool_result for EVERY
            # toolUse block, otherwise the Converse API raises a
            # ValidationException about missing tool_result blocks.
            tool_results = []
            for block in assistant_message["content"]:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    if tool_use["name"] == "calculate_wer":
                        tool_input = tool_use["input"]

                        # Execute programmatic WER calculation
                        result = _calculate_wer(
                            substitutions=tool_input.get("substitutions", 0),
                            deletions=tool_input.get("deletions", 0),
                            insertions=tool_input.get("insertions", 0),
                            reference_words=tool_input.get("reference_words", 1),
                        )
                        result["normalized_reference"] = tool_input.get("normalized_reference")
                        result["normalized_hypothesis"] = tool_input.get("normalized_hypothesis")
                        result["errors"] = tool_input.get("errors", [])

                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"json": result}],
                            }
                        })
                    else:
                        # Unknown tool — still must return a tool_result to
                        # satisfy the Converse API protocol requirement.
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"json": {
                                    "error": f"Unknown tool '{tool_use['name']}'"
                                }}],
                                "status": "error",
                            }
                        })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                reasoning_trace.append({
                    "role": "user",
                    "content": tool_results,
                })

            # If we got a result, get final response then break
            if result is not None:
                try:
                    final_payload = {
                        "modelId": SEMANTIC_WER_MODEL,
                        "system": system_prompt,
                        "messages": messages,
                        "toolConfig": tool_config,
                        "inferenceConfig": {
                            "maxTokens": 1024,
                            "temperature": SEMANTIC_WER_TEMPERATURE,
                        },
                    }
                    if "sonnet-4" in SEMANTIC_WER_MODEL or "claude-sonnet" in SEMANTIC_WER_MODEL:
                        final_payload["additionalModelRequestFields"] = {
                            "thinking": {"type": "disabled"}
                        }
                    final_response = _converse_with_retry(client, final_payload)
                    final_message = final_response["output"]["message"]
                    reasoning_trace.append({
                        "role": "assistant",
                        "content": final_message["content"],
                        "stop_reason": final_response["stopReason"],
                    })
                except Exception as e:
                    logger.warning("[SemWER] Error getting final response: %s", e)
                break

    duration_ms = int((time.time() - start_time) * 1000)

    if result is None:
        raise RuntimeError(
            f"Semantic WER evaluation failed: model did not call calculate_wer "
            f"after {num_turns} turns"
        )

    return SemanticWERResult(
        wer=result["wer"],
        substitutions=result["substitutions"],
        deletions=result["deletions"],
        insertions=result["insertions"],
        reference_words=result["reference_words"],
        total_errors=result["total_errors"],
        normalized_reference=result.get("normalized_reference"),
        normalized_hypothesis=result.get("normalized_hypothesis"),
        errors=result.get("errors", []),
        reasoning_trace=reasoning_trace,
        num_turns=num_turns,
        duration_ms=duration_ms,
    )


# ── Public API (backward-compatible) ───────────────────────────────────────


def compute_semantic_wer(
    reference_text: str,
    hypothesis_text: str,
) -> Optional[float]:
    """Compute Semantic WER using Pipecat's multi-turn tool-use approach.

    This is the backward-compatible entry point that returns just the WER float.
    For full results including reasoning trace, use evaluate_semantic_wer() directly.

    Args:
        reference_text: Ground truth transcription.
        hypothesis_text: ASR engine hypothesis transcription.

    Returns:
        Semantic WER (0.0 = perfect, 1.0 = total loss), or None on failure.
    """
    if not reference_text or not hypothesis_text:
        return None

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            result = evaluate_semantic_wer(reference_text, hypothesis_text)
            return round(result.wer, 4)
        except Exception as e:
            error_msg = str(e)
            # Retry on ValidationException (message ordering issues) and
            # RuntimeError (model didn't call tool within max turns)
            is_retryable = (
                "ValidationException" in error_msg
                or "did not call calculate_wer" in error_msg
            )
            if is_retryable and attempt < max_attempts:
                logger.warning(
                    "[SemWER] Attempt %d/%d failed: %s — retrying...",
                    attempt, max_attempts, error_msg,
                )
                time.sleep(2 * attempt)
                continue
            logger.error("[SemWER] Evaluation failed: %s", e)
            return None
