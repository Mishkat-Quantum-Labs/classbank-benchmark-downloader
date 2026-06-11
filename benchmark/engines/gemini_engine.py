"""Gemini STT Engine — supports 2.5 Pro, 3.1 Pro, and 3.5 Flash.

Adapted from Origin-Tech's stt_service.py for standalone benchmarking.
Uses the unified get_chat_model() factory from llm_client (langchain init_chat_model)
for the LLM call, and google-genai SDK for audio file upload/polling.
"""

import logging
import os
import re
import time

from json_repair import repair_json

from benchmark.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_MAP,
    GEMINI_STT_HTTP_TIMEOUT,
    GEMINI_STT_POLL_INTERVAL,
    GEMINI_STT_POLL_TIMEOUT,
    GEMINI_STT_UPLOAD_RETRIES,
    GEMINI_STT_UPLOAD_RETRY_DELAY,
    RESULTS_DIR,
)
from benchmark.llm_client import get_chat_model
from benchmark.prompts import build_transcription_prompt

logger = logging.getLogger("benchmark")


# ── Error classification ────────────────────────────────────────────────────


def _is_non_retryable(exc: Exception) -> bool:
    """Return True for errors that will never succeed on retry."""
    msg = str(exc)
    if "spending cap" in msg.lower() or "spend cap" in msg.lower():
        return True
    if "401" in msg or "403" in msg or "PERMISSION_DENIED" in msg or "UNAUTHENTICATED" in msg:
        return True
    if "400" in msg or "INVALID_ARGUMENT" in msg:
        return True
    return False


# ── Speaker label normalization ─────────────────────────────────────────────

_STANDARD_PATTERN = re.compile(r"^(Teacher|Observer|Student \d+)$")


def _normalize_speaker_labels(speakers: list[dict]) -> dict[int, str]:
    """Normalize Gemini speaker labels to standardized roles.

    Ensures every speaker gets a clean label: Teacher, Observer, or Student N.
    """
    voice_to_name: dict[int, str] = {}
    student_counter = 0
    observer_counter = 0
    teacher_assigned = False

    for s in speakers:
        voice = s.get("voice", 0)
        name = s.get("name", "").strip()
        role = s.get("role", "").strip().lower()

        # If Gemini already returned a proper standardized label, keep it
        if _STANDARD_PATTERN.match(name):
            voice_to_name[voice] = name
            if name == "Teacher":
                teacher_assigned = True
            elif name.startswith("Student"):
                num = int(name.split()[-1])
                student_counter = max(student_counter, num)
            elif name.startswith("Observer"):
                parts = name.split()
                if len(parts) > 1:
                    observer_counter = max(observer_counter, int(parts[-1]))
                else:
                    observer_counter = max(observer_counter, 1)
            continue

        # Fix based on role
        if role == "teacher" and not teacher_assigned:
            voice_to_name[voice] = "Teacher"
            teacher_assigned = True
        elif role == "observer":
            observer_counter += 1
            voice_to_name[voice] = (
                f"Observer {observer_counter}" if observer_counter > 1 else "Observer"
            )
        elif role == "student":
            student_counter += 1
            voice_to_name[voice] = f"Student {student_counter}"
        else:
            # Fallback: assign as student
            student_counter += 1
            voice_to_name[voice] = f"Student {student_counter}"
            logger.warning(
                "Unrecognized speaker label '%s' role '%s' for voice %d — defaulting to Student %d",
                name, role, voice, student_counter,
            )

    return voice_to_name


# ── Gemini response schema ──────────────────────────────────────────────────

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "task1_transcripts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Start time in seconds"},
                    "end": {"type": "number", "description": "End time in seconds"},
                    "text": {"type": "string"},
                    "voice": {"type": "integer"},
                },
                "required": ["start", "end", "text", "voice"],
            },
        },
        "task2_speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "voice": {"type": "integer"},
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["voice", "name"],
            },
        },
        "detected_language": {
            "type": "string",
            "description": "ISO 639-1 two-letter code of the primary language spoken",
        },
    },
    "required": ["task1_transcripts", "task2_speakers", "detected_language"],
}


# ── Content extraction (Gemini 3+ list-based format) ───────────────────────


def _extract_text_from_content(content) -> str:
    """Extract plain text from LangChain response.content.

    langchain-google-genai v4+ returns different formats depending on model version:
      - Gemini 2.x: str (e.g. '{"task1_transcripts": [...]}')
      - Gemini 3.x: list of content blocks
        [{"type": "text", "text": "..."}, {"type": "thinking", "thinking": "..."}]

    This function normalizes both to a plain text string, extracting only
    "text" type blocks and ignoring "thinking" blocks.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
                # Skip "thinking" blocks — they contain reasoning, not the JSON response
        return "".join(text_parts)

    # Fallback: coerce to string
    return str(content)


class GeminiEngine:
    """Transcription engine using Google Gemini models via unified LLM factory."""

    def __init__(self, engine_key: str = "gemini_pro"):
        """Initialize with a specific Gemini model variant.

        Args:
            engine_key: One of "gemini_pro", "gemini_31_pro", "gemini_35_flash".
        """
        if engine_key not in GEMINI_MODEL_MAP:
            raise ValueError(
                f"Unknown Gemini engine '{engine_key}'. "
                f"Available: {list(GEMINI_MODEL_MAP.keys())}"
            )
        self.engine_key = engine_key
        self.model_id = GEMINI_MODEL_MAP[engine_key]

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file using Gemini via LangChain init_chat_model.

        Uses google-genai SDK for file upload/polling (required for large audio),
        then the unified get_chat_model() factory for the LLM call.

        Args:
            audio_path: Local path to the audio file (MP3, MP4, WAV, etc.).

        Returns:
            dict with keys: language, speakers, dialogue, dialogue_lines,
            word_count, stt_time, segments
        """
        from google import genai
        from langchain_core.messages import HumanMessage

        t0 = time.time()

        # Determine MIME type from extension
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {
            ".m4a": "audio/mp4",
            ".mp4": "audio/mp4",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".opus": "audio/opus",
        }
        mime_type = mime_map.get(ext, "audio/mp4")

        # Initialize google-genai client for file upload only
        http_opts = {"timeout": GEMINI_STT_HTTP_TIMEOUT}
        genai_client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)

        # Upload audio file via google-genai (LangChain doesn't handle file upload)
        audio_file = self._upload_file(genai_client, audio_path, mime_type)

        # Poll until file is ready
        audio_file = self._wait_for_processing(genai_client, audio_file)

        # Get LLM via unified factory
        llm = get_chat_model(self.engine_key)

        # Build multimodal message with file URI + transcription prompt
        prompt = build_transcription_prompt()
        message = HumanMessage(
            content=[
                {
                    "type": "media",
                    "mime_type": mime_type,
                    "file_uri": audio_file.uri,
                },
                {"type": "text", "text": prompt},
            ]
        )

        logger.info("[Gemini] Calling get_chat_model('%s') via init_chat_model…", self.engine_key)

        # Retry up to 3 attempts for transient errors or invalid JSON
        max_attempts = 3
        parsed = None
        raw_text = ""

        for attempt in range(1, max_attempts + 1):
            try:
                response = llm.invoke(
                    [message],
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                )
                logger.info(
                    "[Gemini] LLM invoke returned successfully (engine=%s, attempt=%d/%d)",
                    self.engine_key, attempt, max_attempts,
                )
                raw_content = response.content

                # langchain-google-genai v4+ returns list-based content blocks
                # for Gemini 3+ models (gemini-3.1-pro, gemini-3.5-flash).
                # Format: [{"type": "text", "text": "..."}, ...]
                # For Gemini 2.x models it returns a plain string.
                raw_text = _extract_text_from_content(raw_content)

                logger.debug(
                    "[Gemini] response.content type=%s, extracted text length=%d, "
                    "first 300 chars: %s",
                    type(raw_content).__name__, len(raw_text), repr(raw_text[:300]),
                )

                # Handle empty response
                if not raw_text.strip():
                    logger.warning(
                        "[Gemini] Empty response from model (attempt %d/%d, engine=%s)",
                        attempt, max_attempts, self.engine_key,
                    )
                    parsed = None
                    if attempt < max_attempts:
                        time.sleep(5 * attempt)
                    continue

                # Try to repair and parse JSON
                parsed = repair_json(raw_text, return_objects=True)

                if isinstance(parsed, dict):
                    # Validate expected keys exist
                    if "task1_transcripts" not in parsed:
                        logger.warning(
                            "[Gemini] Response parsed as dict but missing 'task1_transcripts' "
                            "(attempt %d/%d). Keys found: %s",
                            attempt, max_attempts, list(parsed.keys()),
                        )
                        parsed = None
                        if attempt < max_attempts:
                            time.sleep(5 * attempt)
                        continue
                    break

                # Log what we got instead
                logger.warning(
                    "[Gemini] Invalid JSON response (attempt %d/%d, engine=%s). "
                    "Parsed type=%s. Raw (first 500 chars): %s",
                    attempt, max_attempts, self.engine_key,
                    type(parsed).__name__, raw_text[:500],
                )
                parsed = None
                if attempt < max_attempts:
                    time.sleep(5 * attempt)

            except Exception as e:
                if _is_non_retryable(e):
                    raise
                logger.warning(
                    "[Gemini] Transcription error (attempt %d/%d): %s",
                    attempt, max_attempts, e,
                )
                if attempt == max_attempts:
                    raise
                time.sleep(5 * attempt)

        if not isinstance(parsed, dict):
            # Dump last failed response to file for debugging
            debug_path = RESULTS_DIR / f"debug_{self.engine_key}_last_failed_response.txt"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(f"Engine: {self.engine_key}\n")
                df.write(f"Model: {self.model_id}\n")
                df.write(f"Audio: {audio_path}\n")
                df.write(f"Response length: {len(raw_text)}\n")
                df.write(f"Response type after repair_json: {type(parsed).__name__}\n")
                df.write("=" * 60 + "\n")
                df.write(raw_text)
            logger.error(
                "[Gemini] Failed response dumped to: %s", debug_path,
            )
            raise RuntimeError(
                f"Gemini returned invalid JSON after {max_attempts} attempts. "
                f"Last response (first 500 chars): {raw_text[:500]}"
            )

        stt_time = time.time() - t0

        # Extract and normalize
        transcripts = parsed.get("task1_transcripts", [])
        speakers = parsed.get("task2_speakers", [])
        voice_to_name = _normalize_speaker_labels(speakers)
        detected_lang = parsed.get("detected_language", "unknown")
        speaker_count = len(speakers)

        # Build dialogue and segments
        lines = []
        raw_segments = []
        for t in transcripts:
            voice = t.get("voice", 0)
            speaker_name = voice_to_name.get(voice, f"Speaker {voice}")
            start_sec = float(t.get("start", 0))
            end_sec = float(t.get("end", 0))
            text = t.get("text", "")

            minutes = int(start_sec // 60)
            seconds = int(start_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
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

        # Cleanup uploaded file
        try:
            genai_client.files.delete(name=audio_file.name)
        except Exception:
            pass

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

    def _upload_file(self, client, audio_path: str, mime_type: str):
        """Upload audio file to Gemini with retries."""
        audio_file = None
        for attempt in range(GEMINI_STT_UPLOAD_RETRIES):
            try:
                audio_file = client.files.upload(
                    file=audio_path, config={"mime_type": mime_type}
                )
                break
            except Exception as e:
                if _is_non_retryable(e):
                    raise
                if attempt == GEMINI_STT_UPLOAD_RETRIES - 1:
                    raise RuntimeError(
                        f"Gemini upload failed after {GEMINI_STT_UPLOAD_RETRIES} attempts: {e}"
                    ) from e
                time.sleep(GEMINI_STT_UPLOAD_RETRY_DELAY)
        return audio_file

    def _wait_for_processing(self, client, audio_file):
        """Poll until file processing completes."""
        poll_start = time.time()
        while audio_file.state.name == "PROCESSING":
            elapsed = time.time() - poll_start
            if elapsed > GEMINI_STT_POLL_TIMEOUT:
                raise RuntimeError(
                    f"Gemini file processing timed out after {int(elapsed)}s"
                )
            logger.info(
                "[Gemini] Waiting for file processing… state=%s, elapsed=%ds",
                audio_file.state.name,
                int(elapsed),
            )
            time.sleep(GEMINI_STT_POLL_INTERVAL)
            try:
                audio_file = client.files.get(name=audio_file.name)
            except Exception as e:
                if _is_non_retryable(e):
                    raise
                logger.warning("[Gemini] Poll error: %s", e)

        if audio_file.state.name != "ACTIVE":
            raise RuntimeError(
                f"Gemini file processing ended with state: {audio_file.state.name}"
            )
        return audio_file
