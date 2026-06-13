"""DiCoW (Diarization-Conditioned Whisper) STT Engine.

Uses DiCoW v3 MLC for target-speaker ASR with integrated diarization
via DiariZen. This model conditions Whisper on speaker diarization outputs
for accurate multi-speaker transcription.

Model: BUT-FIT/DiCoW_v3_MLC
Base: Whisper large-v3 (fine-tuned with diarization conditioning)
Architecture: Whisper encoder-decoder + FDDT layers + DiariZen diarization
Languages: Same as Whisper large-v3 (99 languages, code-switching capable)
License: Check model card

Diarization: BUT-FIT/diarizen-wavlm-large-s80-md (built-in)

NOTE: DiCoW integrates diarization directly into the ASR decoding process
rather than applying it post-hoc. This means speaker information guides
the transcription, producing per-speaker outputs natively.

Requires: transformers, torch, torchaudio, diarizen
Install DiCoW dependencies: pip install diarizen
"""

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("benchmark")

# Model identifiers
DICOW_MODEL = "BUT-FIT/DiCoW_v3_MLC"
DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"


class DiCoWEngine:
    """Transcription engine using DiCoW (Diarization-Conditioned Whisper).

    Pipeline:
    1. DiariZen: Speaker diarization (who spoke when)
    2. DiCoW: Diarization-conditioned ASR (transcribes per-speaker)

    Unlike WhisperX which applies diarization post-hoc, DiCoW uses
    diarization as a conditioning signal during Whisper decoding.
    This produces more accurate speaker-attributed transcripts,
    especially for overlapping speech.

    Supports code-switching since it's built on Whisper large-v3.
    """

    def __init__(
        self,
        device: str = "cpu",
        dicow_model: str = DICOW_MODEL,
        diarization_model: str = DIARIZEN_MODEL,
    ):
        """Initialize the DiCoW engine.

        Args:
            device: "cuda" for GPU, "cpu" for CPU inference.
            dicow_model: HuggingFace model ID for DiCoW.
            diarization_model: HuggingFace model ID for DiariZen diarization.
        """
        self.device = device
        self.dicow_model_name = dicow_model
        self.diarization_model_name = diarization_model

        self._dicow_model = None
        self._feature_extractor = None
        self._tokenizer = None
        self._diar_pipeline = None
        self._pipeline = None

    def _load_models(self):
        """Lazy-load DiCoW and DiariZen models on first use."""
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoFeatureExtractor,
            AutoTokenizer,
        )

        if self._pipeline is not None:
            return

        # Determine device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[DiCoW] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        # Load DiCoW model
        logger.info("[DiCoW] Loading ASR model: %s (device=%s)", self.dicow_model_name, self.device)

        self._dicow_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.dicow_model_name,
            trust_remote_code=True,
        )
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.dicow_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.dicow_model_name)

        # Set tokenizer on model (required by DiCoW)
        self._dicow_model.set_tokenizer(self._tokenizer)

        # Move to device
        self._dicow_model = self._dicow_model.to(self.device)

        logger.info("[DiCoW] ASR model loaded successfully.")

        # Load DiariZen diarization pipeline
        logger.info("[DiCoW] Loading diarization model: %s", self.diarization_model_name)

        from diarizen import DiariZenPipeline

        self._diar_pipeline = DiariZenPipeline.from_pretrained(
            self.diarization_model_name
        ).to(self.device)

        logger.info("[DiCoW] Diarization model loaded successfully.")

        # Create the combined pipeline
        from diarizen import DiCoWPipeline

        self._pipeline = DiCoWPipeline(
            self._dicow_model,
            diarization_pipeline=self._diar_pipeline,
            feature_extractor=self._feature_extractor,
            tokenizer=self._tokenizer,
            device=self.device,
        )

        logger.info("[DiCoW] Combined pipeline initialized successfully.")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file with diarization-conditioned ASR.

        Args:
            audio_path: Local path to the audio file (WAV recommended, also
                       supports MP3, MP4, FLAC via ffmpeg).

        Returns:
            dict with keys: language, speakers, dialogue, dialogue_lines,
            word_count, stt_time, segments
        """
        import torch
        import torchaudio

        t0 = time.time()

        # Lazy-load models
        self._load_models()

        # ── Step 1: Run DiCoW pipeline ──────────────────────────────────────
        logger.info("[DiCoW] Processing: %s", audio_path)

        # DiCoW pipeline handles both diarization and ASR internally
        with torch.no_grad():
            result = self._pipeline(audio_path)

        # ── Step 2: Parse output ────────────────────────────────────────────
        # DiCoW returns per-speaker transcriptions with timestamps
        # Output format: list of segments with speaker, start, end, text
        segments = self._parse_pipeline_output(result)

        # ── Step 3: Build output ────────────────────────────────────────────
        unique_speakers = []
        for seg in segments:
            if seg["speaker"] not in unique_speakers:
                unique_speakers.append(seg["speaker"])

        speaker_count = len(unique_speakers)

        # Build speaker label map
        speaker_name_map = {}
        for i, spk in enumerate(unique_speakers, start=1):
            speaker_name_map[spk] = f"Speaker {i}"

        lines = []
        raw_segments = []
        for seg in segments:
            start_sec = seg["start"]
            end_sec = seg["end"]
            text = seg["text"].strip()
            speaker_label = speaker_name_map.get(seg["speaker"], "Unknown")

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

        # Detect language from first segment or default
        detected_language = self._detect_language(segments)

        return {
            "language": detected_language,
            "speakers": speaker_count,
            "dialogue": dialogue,
            "dialogue_lines": len(lines),
            "word_count": len(dialogue.split()),
            "stt_time": round(stt_time, 3),
            "segments": raw_segments,
        }

    def _parse_pipeline_output(self, result) -> list[dict]:
        """Parse DiCoW pipeline output into standardized segments.

        The pipeline output format may vary depending on the version.
        This method handles multiple possible formats.

        Returns list of dicts with keys: start, end, speaker, text
        """
        segments = []

        if isinstance(result, dict):
            # Format: {"segments": [...], "speakers": [...]}
            if "segments" in result:
                for seg in result["segments"]:
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "speaker": seg.get("speaker", "speaker_0"),
                        "text": seg.get("text", ""),
                    })
            # Format: {"speaker_0": "text...", "speaker_1": "text..."}
            elif all(k.startswith("speaker") for k in result.keys()):
                for speaker, text in result.items():
                    segments.append({
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": speaker,
                        "text": text if isinstance(text, str) else str(text),
                    })

        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    segments.append({
                        "start": float(item.get("start", item.get("start_time", 0))),
                        "end": float(item.get("end", item.get("end_time", 0))),
                        "speaker": item.get("speaker", item.get("speaker_id", "speaker_0")),
                        "text": item.get("text", item.get("transcription", "")),
                    })
                elif isinstance(item, tuple) and len(item) >= 3:
                    # (speaker, start, end, text) or similar
                    segments.append({
                        "start": float(item[1]) if len(item) > 1 else 0.0,
                        "end": float(item[2]) if len(item) > 2 else 0.0,
                        "speaker": str(item[0]),
                        "text": str(item[3]) if len(item) > 3 else "",
                    })

        elif isinstance(result, str):
            # Single string output — parse RTTM-style or plain text
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "speaker": "speaker_0",
                "text": result,
            })

        if not segments:
            logger.warning("[DiCoW] Could not parse pipeline output: %s", type(result))
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "speaker": "speaker_0",
                "text": str(result) if result else "",
            })

        return segments

    def _detect_language(self, segments: list[dict]) -> str:
        """Attempt to detect language from transcription output.

        DiCoW doesn't explicitly output language code, so we use
        the Whisper tokenizer's language detection if available,
        or default to "en".
        """
        # If the tokenizer has language info from decoding, use it
        if hasattr(self._dicow_model, "detected_language"):
            return self._dicow_model.detected_language

        # Default — DiCoW is multilingual but doesn't always expose the detected lang
        return "en"
