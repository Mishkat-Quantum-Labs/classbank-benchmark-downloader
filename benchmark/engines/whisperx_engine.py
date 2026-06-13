"""WhisperX STT Engine — Whisper large-v3 + Pyannote Diarization.

Uses WhisperX for fast batched ASR with word-level timestamps and
speaker diarization via pyannote-audio.

Model: openai/whisper-large-v3 (via faster-whisper backend)
WER: ~7.44% (Open ASR Leaderboard)
Architecture: Whisper encoder-decoder (faster-whisper CTranslate2 backend)
Languages: 99 languages, handles code-switching
License: MIT (WhisperX), Apache-2.0 (Whisper)

Diarization: pyannote/speaker-diarization-community-1 (CC-BY-4.0)
Requires: HF_TOKEN with accepted user agreement for pyannote models.

NOTE: Supports CPU inference. Use compute_type="int8" for CPU,
"float16" for GPU. Handles code-switching naturally since Whisper
was trained on multilingual data with mixed-language utterances.
"""

import logging
import os
import time

logger = logging.getLogger("benchmark")

# Default configuration
DEFAULT_MODEL = "large-v3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_COMPUTE_TYPE_GPU = "float16"
DEFAULT_COMPUTE_TYPE_CPU = "int8"


class WhisperXEngine:
    """Transcription engine using WhisperX (Whisper + forced alignment + Pyannote diarization).

    Pipeline:
    1. ASR: Whisper large-v3 via faster-whisper (batched inference)
    2. Alignment: wav2vec2 forced phoneme alignment for word-level timestamps
    3. Diarization: Pyannote speaker-diarization-community-1
    4. Speaker assignment: WhisperX word-to-speaker mapping

    Supports code-switching (multilingual within same utterance) since
    Whisper was trained on diverse multilingual audio including code-switched speech.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        compute_type: str | None = None,
        hf_token: str | None = None,
    ):
        """Initialize the WhisperX engine.

        Args:
            model_name: Whisper model size. One of "tiny", "base", "small",
                       "medium", "large-v2", "large-v3".
            device: "cuda" for GPU, "cpu" for CPU inference.
            batch_size: Batch size for transcription. Reduce if low on memory.
            compute_type: "float16" for GPU, "int8" for CPU. Auto-detected if None.
            hf_token: HuggingFace token for pyannote diarization model access.
                     Falls back to HF_TOKEN env var if not provided.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type or (
            DEFAULT_COMPUTE_TYPE_GPU if device == "cuda" else DEFAULT_COMPUTE_TYPE_CPU
        )
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        self._asr_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None

    def _load_asr_model(self):
        """Lazy-load the WhisperX ASR model."""
        if self._asr_model is None:
            import whisperx

            logger.info(
                "[WhisperX] Loading ASR model: %s (device=%s, compute_type=%s)",
                self.model_name, self.device, self.compute_type,
            )
            self._asr_model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
            )
            logger.info("[WhisperX] ASR model loaded successfully.")

    def _load_diarize_model(self):
        """Lazy-load the pyannote diarization pipeline."""
        if self._diarize_model is None:
            from whisperx.diarize import DiarizationPipeline

            if not self.hf_token:
                raise ValueError(
                    "HF_TOKEN is required for speaker diarization. "
                    "Set it as an environment variable or pass hf_token to the constructor. "
                    "You also need to accept the pyannote model agreement at: "
                    "https://huggingface.co/pyannote/speaker-diarization-community-1"
                )

            logger.info("[WhisperX] Loading diarization model (pyannote)...")
            self._diarize_model = DiarizationPipeline(
                token=self.hf_token,
                device=self.device,
            )
            logger.info("[WhisperX] Diarization model loaded successfully.")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file with speaker diarization.

        Args:
            audio_path: Local path to the audio file (WAV, MP3, MP4, FLAC, etc.).

        Returns:
            dict with keys: language, speakers, dialogue, dialogue_lines,
            word_count, stt_time, segments
        """
        import whisperx

        t0 = time.time()

        # Lazy-load models
        self._load_asr_model()
        self._load_diarize_model()

        # ── Step 1: Transcribe with Whisper (batched) ───────────────────────
        logger.info("[WhisperX] Transcribing: %s", audio_path)

        audio = whisperx.load_audio(audio_path)
        result = self._asr_model.transcribe(audio, batch_size=self.batch_size)

        detected_language = result.get("language", "en")
        logger.info("[WhisperX] Detected language: %s", detected_language)

        # ── Step 2: Align (word-level timestamps) ───────────────────────────
        logger.info("[WhisperX] Running forced alignment...")

        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        except Exception as e:
            # Alignment may fail for unsupported languages — proceed without it
            logger.warning(
                "[WhisperX] Alignment failed for language '%s': %s. "
                "Proceeding with segment-level timestamps.",
                detected_language, e,
            )

        # ── Step 3: Speaker diarization ─────────────────────────────────────
        logger.info("[WhisperX] Running speaker diarization...")

        diarize_segments = self._diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # ── Step 4: Build output ────────────────────────────────────────────
        segments = result.get("segments", [])

        # Collect unique speakers
        unique_speakers = []
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            if speaker not in unique_speakers:
                unique_speakers.append(speaker)

        speaker_count = len(unique_speakers)

        # Build speaker label map (SPEAKER_00 → Speaker 1, etc.)
        speaker_name_map = {}
        for i, spk in enumerate(unique_speakers, start=1):
            speaker_name_map[spk] = f"Speaker {i}"

        lines = []
        raw_segments = []
        for seg in segments:
            start_sec = seg.get("start", 0.0)
            end_sec = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            speaker_id = seg.get("speaker", "UNKNOWN")
            speaker_label = speaker_name_map.get(speaker_id, "Unknown")

            if not text:
                continue

            minutes = int(start_sec // 60)
            seconds = int(start_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            lines.append(f"[{timestamp}] {speaker_label}: {text}")

            raw_segments.append({
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "speaker": speaker_label.lower().replace(" ", "_"),
                "text": text,
            })

        stt_time = time.time() - t0
        dialogue = "\n".join(lines)

        return {
            "language": detected_language,
            "speakers": speaker_count,
            "dialogue": dialogue,
            "dialogue_lines": len(lines),
            "word_count": len(dialogue.split()),
            "stt_time": round(stt_time, 3),
            "segments": raw_segments,
        }
