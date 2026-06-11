"""Semantic WER — LLM-as-judge using Claude Sonnet 4.5 via AWS Bedrock.

Same prompt and configuration as Pipecat's STT benchmark,
but routed through Bedrock instead of Anthropic API directly.
https://github.com/pipecat-ai/stt-benchmark
"""

import json
import logging
from typing import Optional

from benchmark.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BEDROCK_REGION
from benchmark.prompts import SEMANTIC_WER_PROMPT

logger = logging.getLogger("benchmark")

# Pipecat config: Claude Sonnet 4.5, temperature=0, max_tokens=4096
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MAX_TOKENS = 4096
TEMPERATURE = 0.0


def compute_semantic_wer(
    reference_text: str,
    hypothesis_text: str,
) -> Optional[float]:
    """Compute Semantic WER using Claude Sonnet 4.5 via Bedrock.

    Same model/prompt/config as Pipecat's benchmark.
    Falls back to heuristic if Bedrock credentials are not configured.

    Returns:
        Semantic WER (0.0 = perfect, 1.0 = total loss), or None on failure.
    """
    if not reference_text or not hypothesis_text:
        return None

    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        try:
            return _bedrock_semantic_wer(reference_text, hypothesis_text)
        except Exception as e:
            logger.warning("[SemWER] Bedrock eval failed: %s — using heuristic", e)

    return _heuristic_semantic_wer(reference_text, hypothesis_text)


def _bedrock_semantic_wer(reference_text: str, hypothesis_text: str) -> Optional[float]:
    """Evaluate with Claude Sonnet 4.5 on Bedrock."""
    import boto3

    max_chars = 3000
    ref = reference_text[:max_chars]
    hyp = hypothesis_text[:max_chars]

    prompt = SEMANTIC_WER_PROMPT.format(reference=ref, hypothesis=hyp)

    client = boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    )

    raw = response["output"]["message"]["content"][0]["text"]

    # Parse JSON from response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(raw[start:end])
        s = data.get("substitutions", 0)
        d = data.get("deletions", 0)
        i = data.get("insertions", 0)
        n = data.get("reference_words", 1)
        if n == 0:
            return 0.0
        return round((s + d + i) / n, 4)

    return None


def _heuristic_semantic_wer(reference_text: str, hypothesis_text: str) -> float:
    """Fallback when no credentials: word overlap heuristic."""
    from benchmark.evaluation import normalize_text

    ref = normalize_text(reference_text).split()
    hyp = normalize_text(hypothesis_text).split()

    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0

    ref_set = set(ref)
    hyp_set = set(hyp)
    unigram_recall = len(ref_set & hyp_set) / len(ref_set)

    ref_bigrams = set(zip(ref[:-1], ref[1:])) if len(ref) > 1 else set()
    hyp_bigrams = set(zip(hyp[:-1], hyp[1:])) if len(hyp) > 1 else set()
    bigram_recall = (
        len(ref_bigrams & hyp_bigrams) / len(ref_bigrams)
        if ref_bigrams else unigram_recall
    )

    score = 0.5 * unigram_recall + 0.5 * bigram_recall
    return round(1.0 - score, 4)
