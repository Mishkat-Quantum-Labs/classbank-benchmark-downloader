"""Prompts for STT engines — transcription, speaker classification, and evaluation."""

# ── Gemini Transcription Prompt ─────────────────────────────────────────────

GEMINI_TRANSCRIPTION_PROMPT = """
**Context:** This is an AUDIO-ONLY classroom recording. There is NO video. Do NOT reference or infer any visual information.

**Task 1 - Transcripts**

- Listen carefully to the audio.
- Identify the distinct voices using a `voice` ID (1, 2, 3, etc.).
- Transcribe the audio verbatim with voice diarization.
- Include the `start` time (in seconds as a decimal number, e.g. 3.5) and `end` time (in seconds as a decimal number, e.g. 7.2) for each speech segment.
- Transcribe ALL speech in its original language and script. Do NOT translate.
- **URDU RULE (HARD REQUIREMENT):** If the spoken language is Hindi, Urdu, or Hindustani, you MUST ALWAYS transcribe in Urdu using the native Urdu script (نستعلیق / Arabic-derived script). NEVER use Devanagari. NEVER use Roman/Latin transliteration. This applies even if the audio sounds closer to Hindi — always choose Urdu script.

**Transcription Rules (HARD REQUIREMENTS):**
- Produce clean, readable transcription focused on core meaning.
- Omit filler words (um, uh, hmm, mhm, uh-huh, ah) and stutters (th-that, b-but). Omit false starts and self-corrections — output only the speaker's intended final phrasing.
- Add standard punctuation (periods, commas, question marks) based on speech prosody and intonation.
- Write numbers as spoken words (e.g., "fifteen" not "15", "two thousand twenty-four" not "2024").
- If speech is completely inaudible or unintelligible, skip that portion entirely. Do NOT guess or fabricate words.
- For overlapping speech: transcribe the dominant/clearer speaker only. Attribute the segment to whoever is more audible.
- Segment boundaries: start a new segment when the speaker changes OR when there is a silence gap of 2+ seconds within the same speaker's turn.
- Ignore non-speech sounds (bells, laughter, coughing, door slams, background noise). Do NOT include audio event markers.

**Task 2 - Speaker Identification & Role-Based Labelling**

This is a classroom recording. Every speaker MUST be classified into one of these roles and labelled accordingly:

- **Teacher** — the person leading instruction, explaining concepts, asking questions to the class, giving directions. There is typically ONE teacher. Use the label `Teacher`.
- **Observer** — a person present but NOT participating in instruction (e.g., a school administrator, evaluator, or visitor who makes brief comments but does not teach or learn). Use the label `Observer`. If there are multiple observers, use `Observer 1`, `Observer 2`, etc.
- **Student** — anyone responding to the teacher, asking questions as a learner, or participating as a student. Label them `Student 1`, `Student 2`, `Student 3`, etc., numbered in the order they FIRST speak.

**How to determine roles from AUDIO ONLY:**
- The Teacher is typically the speaker who talks the most, gives instructions, explains concepts, and asks questions directed at the group.
- Students typically give shorter responses, answer questions, or ask clarifying questions.
- An Observer speaks rarely and their speech is not part of the lesson flow (e.g., side comments, greetings to the teacher).
- Use speech patterns, tone of authority, and conversational dynamics — NOT visual cues.

**LABELLING RULES (HARD REQUIREMENT):**
- The `name` field MUST be the standardized label: `Teacher`, `Observer`, `Student 1`, `Student 2`, etc.
- Do NOT use real names, generic labels like `Speaker 1`, or `?` in the `name` field.
- The `role` field MUST be one of: `teacher`, `observer`, `student`.
- If you cannot confidently identify a speaker as Teacher or Observer, default to `Student N`.
- Student numbering is by order of first appearance in the audio.

**Task 3 - Language Detection**

- Identify the primary language spoken in the audio.
- Return its ISO 639-1 two-letter code in the `detected_language` field (e.g., "ur", "en", "ar").
- If the language is Hindi or Hindustani, report it as `"ur"` (Urdu).
"""

# ── Speaker Classification Prompt (ElevenLabs pipeline) ─────────────────────

SPEAKER_CLASSIFICATION_PROMPT = """You are analyzing a classroom audio transcript. There are {speaker_count} speakers labeled "Speaker 1", "Speaker 2", etc.

Your task: Classify each speaker into exactly ONE role based on their speech patterns.

**Roles:**
- **Teacher** — the person leading instruction, explaining concepts, asking questions to the class, giving directions. There is typically ONE teacher.
- **Observer** — a person present but NOT participating in instruction (e.g., school administrator, evaluator who makes brief comments but does not teach or learn). Rare.
- **Student** — anyone responding to the teacher, asking questions as a learner, or participating as a student.

**How to determine roles:**
- The Teacher typically talks the most, gives instructions, explains concepts, and asks questions directed at the group.
- Students typically give shorter responses, answer questions, or ask clarifying questions.
- An Observer speaks rarely and their speech is not part of the lesson flow.

**Output format (JSON only, no explanation):**
{{
  "Speaker 1": "Teacher",
  "Speaker 2": "Student 1",
  "Speaker 3": "Student 2"
}}

Rules:
- Exactly ONE Teacher (the speaker who leads instruction).
- Students numbered in order: Student 1, Student 2, etc.
- Only assign Observer if a speaker clearly does not participate in the lesson.
- If unsure, default to Student.

**Transcript:**
{transcript}"""


def build_transcription_prompt() -> str:
    """Return the Gemini multimodal transcription prompt."""
    return GEMINI_TRANSCRIPTION_PROMPT


def build_speaker_classification_prompt(transcript: str, speaker_count: int) -> str:
    """Build the LLM prompt for classifying speakers as Teacher/Observer/Student."""
    return SPEAKER_CLASSIFICATION_PROMPT.format(
        speaker_count=speaker_count,
        transcript=transcript,
    )


# ── Semantic WER Prompt (LLM-as-judge) ──────────────────────────────────────
# Adapted from Pipecat's open-source STT benchmark (MIT license)
# https://github.com/pipecat-ai/stt-benchmark

SEMANTIC_WER_PROMPT = """You are an expert ASR evaluator. Calculate the Semantic Word Error Rate (WER) - counting ONLY transcription errors that would impact meaning.

## Rules

Count as ERROR:
- Word substitutions that change meaning ("card" → "car", "trace" → "trade")
- Nonsense/hallucinated words ("lentil" → "landon")
- Missing words that change intent
- Wrong names, numbers, negations

Do NOT count as error:
- Punctuation, capitalization differences
- Contractions ("don't" = "do not")
- Singular/plural when meaning is preserved ("license" = "licenses")
- Number format ("3" = "three", "$5" = "five dollars")
- Filler words (um, uh, like)
- Missing articles ("the", "a") that don't change meaning
- British/American spelling ("colour" = "color")
- Hyphenation ("long-term" = "long term")
- Spoken variations ("gonna" = "going to", "yeah" = "yes")
- Possessives ("driver's" = "drivers" = "driver")

## Process

1. Normalize both texts (lowercase, expand contractions, remove fillers)
2. Align word-by-word
3. For each difference, ask: "Would an LLM interpret these differently?"
4. Count ONLY differences where the answer is YES

## Output

Reply with ONLY a JSON object:
{{"substitutions": S, "deletions": D, "insertions": I, "reference_words": N}}

Where:
- S = words that changed meaning
- D = words lost that change meaning
- I = extra words that add wrong meaning
- N = total words in normalized reference

REFERENCE:
{reference}

HYPOTHESIS:
{hypothesis}"""


def build_semantic_wer_prompt(reference: str, hypothesis: str) -> str:
    """Build the LLM prompt for semantic WER evaluation."""
    return SEMANTIC_WER_PROMPT.format(reference=reference, hypothesis=hypothesis)
