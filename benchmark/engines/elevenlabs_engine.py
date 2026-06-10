"""ElevenLabs Scribe v2 STT Engine.

Adapted from Origin-Tech's stt_service.py for standalone benchmarking.
Uses ElevenLabs for transcription + AWS Bedrock Haiku for speaker role classification.
"""

import logging
import time

from benchmark.config import (
    ELEVENLABS_API_KEY,
    SEGMENT_MAX_DURATION_SEC,
    SEGMENT_PAUSE_THRESHOLD_MS,
)
from benchmark.llm_client import classify_speakers

logger = logging.getLogger("benchmark")


# ── Language code normalization ─────────────────────────────────────────────

_LANG_MAP = {
    "eng": "en",
    "urd": "ur",
    "ara": "ar",
    "hin": "hi",
    "fra": "fr",
    "spa": "es",
    "deu": "de",
    "por": "pt",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
    "tur": "tr",
    "fas": "fa",
    "pus": "ps",
}


def _normalize_language_code(code: str) -> str:
    """Convert ISO 639-3 codes to ISO 639-1 (2-letter) codes."""
    if len(code) == 2:
        return code
    return _LANG_MAP.get(code.lower(), code[:2] if len(code) >= 2 else code)


# ── Segment splitting ───────────────────────────────────────────────────────


def _split_long_segments(segments: list[dict], max_duration_sec: float) -> list[dict]:
    """Split segments exceeding max_duration_sec at sentence boundaries."""
    max_duration_ms = max_duration_sec * 1000
    result = []

    for seg in segments:
        duration_ms = seg["end_ms"] - seg["start_ms"]
        if duration_ms <= max_duration_ms:
            result.append(seg)
            continue

        text = seg["text"]
        split_indices = [
            i for i, ch in enumerate(text)
            if ch in ".?!" and i < len(text) - 1
        ]

        if not split_indices:
            mid = len(text) // 2
            space_after = text.find(" ", mid)
            space_before = text.rfind(" ", 0, mid)
            split_at = space_after if space_after != -1 else space_before
            if split_at == -1:
                result.append(seg)
                continue
            split_indices = [split_at]

        halfway = len(text) // 2
        best_split = None
        for idx in split_indices:
            if idx >= halfway:
                best_split = idx + 1
                break
        if best_split is None:
            best_split = split_indices[-1] + 1

        ratio = best_split / len(text) if len(text) > 0 else 0.5
        mid_time_ms = seg["start_ms"] + int(duration_ms * ratio)

        result.append({
            "speaker_id": seg["speaker_id"],
            "text": text[:best_split],
            "start_ms": seg["start_ms"],
            "end_ms": mid_time_ms,
        })
        result.append({
            "speaker_id": seg["speaker_id"],
            "text": text[best_split:],
            "start_ms": mid_time_ms,
            "end_ms": seg["end_ms"],
        })

    return result


class ElevenLabsEngine:
    """Transcription engine using ElevenLabs Scribe v2 + LLM role classification."""

    def __init__(self):
        """Initialize ElevenLabs engine."""
        if not ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY is not configured")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file using ElevenLabs Scribe v2.

        Flow:
        1. ElevenLabs STT → transcription with speaker diarization
        2. Haiku LLM → classify each speaker as Teacher/Observer/Student
        3. Apply role labels → produce final formatted output

        Args:
            audio_path: Local path to the audio file.

        Returns:
            dict with keys: language, speakers, dialogue, dialogue_lines,
            word_count, stt_time, segments
        """
        from elevenlabs.client import ElevenLabs as ElevenLabsClient

        t0 = time.time()

        client = ElevenLabsClient(api_key=ELEVENLABS_API_KEY)

        # Step 1: ElevenLabs transcription with diarization
        with open(audio_path, "rb") as f:
            transcription = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v2",
                diarize=True,
                tag_audio_events=False,
                no_verbatim=True,
                language_code=None,  # auto-detect
            )

        # Parse language
        detected_lang = getattr(transcription, "language_code", "unknown") or "unknown"
        detected_lang = _normalize_language_code(detected_lang)

        # Build segments from word-level output
        words = getattr(transcription, "words", []) or []
        segments: list[dict] = []
        current_segment: dict | None = None

        for word in words:
            speaker_id = getattr(word, "speaker_id", None)
            text = getattr(word, "text", "") or ""
            start_ms = getattr(word, "start", None)
            end_ms = getattr(word, "end", None)

            # Skip words without timestamps
            if start_ms is None or end_ms is None:
                if current_segment and text:
                    current_segment["text"] += text
                continue

            if speaker_id is None:
                if current_segment:
                    current_segment["text"] += text
                    current_segment["end_ms"] = end_ms
                continue

            # Determine if we should start a new segment
            should_split = False
            if current_segment is None:
                should_split = True
            elif current_segment["speaker_id"] != speaker_id:
                should_split = True
            elif (
                SEGMENT_PAUSE_THRESHOLD_MS > 0
                and start_ms - current_segment["end_ms"] >= SEGMENT_PAUSE_THRESHOLD_MS
            ):
                should_split = True

            if should_split:
                if current_segment:
                    segments.append(current_segment)
                current_segment = {
                    "speaker_id": speaker_id,
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            else:
                current_segment["text"] += text
                if end_ms:
                    current_segment["end_ms"] = end_ms

        if current_segment:
            segments.append(current_segment)

        # Force-split long segments at sentence boundaries
        if SEGMENT_MAX_DURATION_SEC > 0:
            segments = _split_long_segments(segments, SEGMENT_MAX_DURATION_SEC)

        # Get unique speaker IDs in order of first appearance
        unique_speakers: list[int] = []
        for seg in segments:
            if seg["speaker_id"] not in unique_speakers:
                unique_speakers.append(seg["speaker_id"])

        speaker_count = len(unique_speakers)

        # Build raw transcript for LLM role classification
        raw_lines = []
        for seg in segments:
            start_sec = seg["start_ms"] / 1000
            minutes = int(start_sec // 60)
            seconds = int(start_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            speaker_label = f"Speaker {unique_speakers.index(seg['speaker_id']) + 1}"
            text = seg["text"].strip()
            if text:
                raw_lines.append(f"[{timestamp}] {speaker_label}: {text}")

        raw_transcript = "\n".join(raw_lines)

        # Step 2: LLM role classification via Haiku
        voice_to_name = classify_speakers(raw_transcript, speaker_count)

        # Map Speaker N labels back to speaker_ids
        speaker_id_to_name: dict[int, str] = {}
        for sid in unique_speakers:
            idx = unique_speakers.index(sid) + 1
            speaker_key = f"Speaker {idx}"
            if speaker_key not in voice_to_name:
                raise RuntimeError(
                    f"LLM role classification missing label for '{speaker_key}'. "
                    f"LLM returned: {voice_to_name}"
                )
            speaker_id_to_name[sid] = voice_to_name[speaker_key]

        # Step 3: Build final dialogue with role labels
        lines = []
        raw_segments = []
        for seg in segments:
            start_sec = seg["start_ms"] / 1000
            end_sec = seg["end_ms"] / 1000
            minutes = int(start_sec // 60)
            seconds = int(start_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            speaker_name = speaker_id_to_name.get(seg["speaker_id"])
            if not speaker_name:
                raise RuntimeError(
                    f"No role label for speaker_id {seg['speaker_id']}. "
                    f"Classification returned: {voice_to_name}"
                )
            text = seg["text"].strip()
            if text:
                lines.append(f"[{timestamp}] {speaker_name}: {text}")
                raw_segments.append({
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "speaker": (
                        speaker_name.lower()
                        if speaker_name in ("Teacher", "Observer")
                        else speaker_name.lower().replace(" ", "_")
                    ),
                    "text": text,
                })

        stt_time = time.time() - t0
        dialogue = "\n".join(lines)

        return {
            "language": detected_lang,
            "speakers": speaker_count,
            "dialogue": dialogue,
            "dialogue_lines": len(lines),
            "word_count": len(dialogue.split()),
            "stt_time": round(stt_time, 3),
            "segments": raw_segments,
        }
