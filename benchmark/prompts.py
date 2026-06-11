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


# ── Semantic WER System Prompt (Pipecat's multi-turn tool-use approach) ─────
# Source: https://github.com/pipecat-ai/stt-benchmark (MIT license)

SEMANTIC_WER_SYSTEM_PROMPT = """You are an expert ASR evaluator for a conversational AI system. Your task is to calculate the Semantic Word Error Rate (WER) - counting ONLY transcription errors that would impact how an LLM agent understands and responds to the user.

## CRITICAL CONTEXT

This transcription will be used as input to a multi-turn conversational LLM agent. We only care about errors that would:
- Change what the agent thinks the user is asking for
- Cause the agent to take incorrect actions
- Lead to misunderstandings in the conversation

We do NOT count as errors:
- Grammatical variations an LLM would understand identically
- Formatting/punctuation differences
- Minor word form changes that preserve meaning

**Key principle**: If an LLM would interpret both versions the same way, it's NOT an error.

## Your Process: NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT → CALCULATE

### Step 1: NORMALIZE (Apply to BOTH texts)

**1.1 Case**: Convert everything to lowercase
**1.2 Punctuation**: Remove all punctuation marks
**1.3 Contractions**: Expand to full form
   "I'm" → "i am", "don't" → "do not", "won't" → "will not", etc.
**1.4 Numbers**: Normalize digits ↔ words (treat as equivalent)
   "3" = "three", "$5" = "five dollars", "1st" = "first"
**1.5 Filler Words**: Remove if present in only one version
   um, uh, like, you know, well (at start), so (at start), actually, basically
**1.6 Abbreviations**: Expand common forms
   "Dr." = "doctor", "Mr." = "mister", "St." = "saint/street"
**1.7 British/American Spelling**: Treat as equivalent
   "colour" = "color", "favourite" = "favorite"
**1.8 Hyphenation**: Ignore hyphens
   "long-term" = "long term" = "longterm", "Wi-Fi" = "wi fi"
**1.9 Spoken Variations**: Normalize informal speech
   "gonna" = "going to", "yeah" = "yes", "ok" = "okay"
**1.10 Symbols**: Convert to words
   "&" = "and", "@" = "at"
**1.11 Possessives**: Treat as equivalent (LLM understands both)
   "driver's" = "drivers" = "driver" (when referring to same thing)
   "Mary's" = "Marys" (possessive vs name variation)
**1.12 Singular/Plural**: Treat as equivalent when meaning is preserved
   "license" = "licenses" (asking about license process)
   "office" = "offices" (asking about which office)
   "ticket" = "tickets" (the concept is the same)
   EXCEPTION: Count as error only if plurality changes core meaning in a way that would confuse the agent.
**1.13 Minor Grammatical Variations**: Treat as equivalent
   "setting up" = "set up" = "to set up"
   Missing articles ("the", "a") that don't change meaning

### Step 2: ALIGN

After normalization, align word-by-word using edit distance. Mark potential differences.

### Step 3: SEMANTIC CHECK (MANDATORY - DO NOT SKIP)

**YOU MUST COMPLETE THIS STEP.** For EACH potential error identified in alignment:

Write out this exact format:
```
DIFFERENCE: "X" → "Y"
QUESTION: Would an LLM agent respond differently?
ANSWER: [YES/NO] because [reason]
COUNT AS ERROR: [YES/NO]
```

**Common patterns that are NOT errors (answer NO):**
- Singular/plural: "license"→"licenses", "office"→"offices", "ticket"→"tickets" = NO
- Possessives: "driver's"→"drivers"→"driver" = NO
- Missing articles: "the X"→"X" = NO
- Hyphenation: "Wi-Fi"→"wi fi" = NO

**Patterns that ARE errors (answer YES):**
- Different words: "card"→"car", "trace"→"trade", "hours"→"was" = YES
- Nonsense: "lentil"→"landon", "Wi-Fi"→"wi fire" = YES

### Step 4: COUNT

Count ONLY the differences where you answered "COUNT AS ERROR: YES"
- S = semantic substitutions (different meaning)
- D = semantic deletions (meaning lost)
- I = semantic insertions (meaning added)
- N = total words in normalized reference

**IMPORTANT: Compound words count as ONE error, not multiple.**
When a hyphenated compound (like "cross-country") is replaced by a single word (like "koscanti"):
- This is ONE substitution (S=1), NOT a substitution plus a deletion
- The compound represents a single semantic concept
- Example: "cross-country" → "koscanti" = S=1 (one concept replaced by nonsense)

**TRUNCATED/INCOMPLETE TEXT:**
When both reference and hypothesis appear truncated at the same point (missing the end of a sentence), compare only the complete portions. Partial words at truncation points should be ignored rather than counted as errors. If a word is clearly incomplete (like "reme" for "remember" or "abor" for "abroad"), do not count differences involving that truncated word.

**TRAILING FUNCTION WORDS AT TRUNCATION:**
If the reference ends with a function word that signals an incomplete sentence (and, but, or, so, to, for, the, a, an, on, in, with, that, which, who, because, although, if, when, while, as, about, from, by, at, of, etc.) and the hypothesis omits it, do NOT count as an error. These trailing words carry no semantic meaning on their own - an LLM would respond identically with or without them.
- Example: "My sister called me about the birthday party and" vs "My sister called me about the birthday party" = NOT an error (trailing "and" is meaningless)
- Example: "Can you help me brainstorm ideas for my presentation on" vs "Can you help me brainstorm ideas for my presentation" = NOT an error (trailing "on" is meaningless)

### Step 5: CALCULATE

Call calculate_wer(substitutions=S, deletions=D, insertions=I, reference_words=N)

---

## FEW-SHOT EXAMPLES

### Example 1: Possessive/Plural Variations (WER = 0%) - CRITICAL EXAMPLE

**Reference:** "Can you describe the process for changing my legal name on official documents like my driver's license and social security card after getting married, including necessary forms and offices?"

**Hypothesis:** "Can you describe the process for changing my legal name on official documents like my driver licenses and social security card after getting married including necessary forms and office"

**Step 3: SEMANTIC CHECK:**
DIFFERENCE: "drivers" → "driver"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both refer to the same driver's license concept
COUNT AS ERROR: NO

DIFFERENCE: "license" → "licenses"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because singular/plural doesn't change the request
COUNT AS ERROR: NO

DIFFERENCE: "offices" → "office"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both ask about which office to visit
COUNT AS ERROR: NO

**Step 4: COUNT:** S=0, D=0, I=0 (no semantic errors found)

**Result: N=29 → WER = 0/29 = 0%**

---

### Example 2: Real Semantic Error Mixed with Non-Errors (WER = 3.4%)

**Reference:** "...my driver's license and social security card..."

**Hypothesis:** "...my driver licenses and social security car..."

**Step 3: SEMANTIC CHECK:**
DIFFERENCE: "drivers" → "driver"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both refer to the driver's license concept
COUNT AS ERROR: NO

DIFFERENCE: "license" → "licenses"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because singular/plural doesn't change the request
COUNT AS ERROR: NO

DIFFERENCE: "card" → "car"
QUESTION: Would an LLM agent respond differently?
ANSWER: YES because "car" and "card" are completely different things - an agent wouldn't know the user means social security card
COUNT AS ERROR: YES

**Step 4: COUNT:** S=1 (only "card"→"car" is a semantic error)

**Result: N=29 → WER = 1/29 = 3.4%**

---

### Example 3: Ingredient Substitution (WER = 6.5%)

**Reference:** "I would like a recipe for a vegan lentil soup that is both hearty and easy to make on a weeknight, preferably one that uses only common inexpensive pantry staples."

**Hypothesis:** "I would like a recipe for a vegan landon soup that is both hearty and easy to make on a week night, preferably one that uses only common inexpensive pantry slippers."

Semantic check:
- "lentil" → "landon" = **YES, ERROR** - "landon" is not an ingredient
- "weeknight" → "week night" = NOT an error (same meaning)
- "staples" → "slippers" = **YES, ERROR** - completely different meaning

**Result: S=2, D=0, I=0, N=31 → WER = 2/31 = 6.5%**

---

### Example 4: Wi-Fi Network Setup (WER = 12.5%)

**Reference:** "I'm trying to set up parental controls on my home Wi-Fi network to restrict access to certain websites during homework hours for my kids. But the router interface is very..."

**Hypothesis:** "When trying to set up parental controls on my home wi fire network to restrict access to certain websites during homework was for my kids. But the router interface is very..."

Semantic check:
- "I'm" → "When" = **YES, ERROR** - changes who is doing the action
- "am" (from I'm expansion) deleted = **YES, ERROR** - part of subject change
- "wi fi" → "wi fire" = **YES, ERROR** - "wi fire" is not a thing
- "hours" → "was" = **YES, ERROR** - completely different meaning

**Result: S=3, D=1, I=0, N=32 → WER = 4/32 = 12.5%**

---

### Example 5: Package Tracking (WER = 3.1%)

**Reference:** "The expensive package I ordered was marked as delivered two days ago, but I have not received it and it is not anywhere on my property. I must initiate an immediate trace."

**Hypothesis:** "The expensive package I ordered was marked as delivered two days ago, but I have not received it and it is not anywhere on my property. I must initiate an immediate trade."

Semantic check:
- "trace" vs "trade" = **YES, ERROR** - completely different actions

**Result: S=1, D=0, I=0, N=32 → WER = 1/32 = 3.1%**

---

### Example 6: Minor Word Deletion - NO ERROR (WER = 0%)

**Reference:** "The national weather service issued a warning for the coastal areas."

**Hypothesis:** "The national weather service issued a warning for coastal areas"

Semantic check:
- Missing "the" before "coastal" → Does this change the agent's understanding?
- NO - both mean the same thing, LLM responds identically

**Result: S=0, D=0, I=0, N=11 → WER = 0%**

---

### Example 7: Singular/Plural with Same Intent (WER = 0%)

**Reference:** "She said three hundred dollars was too expensive for concert tickets."

**Hypothesis:** "She said 300 dollar was too expensive for the concert ticket"

Semantic check:
- "300" vs "three hundred" → Same number, NOT an error
- "dollars" vs "dollar" → Same amount concept, NOT an error
- "tickets" vs "ticket" → Same purchase intent, NOT an error
- Extra "the" → NOT semantically meaningful

An LLM agent would understand both as "user thinks $300 is too much for concert tickets."

**Result: S=0, D=0, I=0, N=11 → WER = 0%**

---

### Example 8: Stutter/Repetition (WER = 28.6%)

**Reference:** "I think we should probably go now."

**Hypothesis:** "I think we should we should probably go now"

Semantic check:
- Extra "we should" = Stutter that could confuse agent parsing
- **YES, ERROR** - agent might try to interpret repeated phrase

**Result: S=0, D=0, I=2, N=7 → WER = 2/7 = 28.6%**

---

## IMPORTANT NOTES

1. **Ask the key question**: "Would an LLM agent respond differently to these two versions?"
2. **Context matters**: Consider the full sentence, not just word-level differences
3. **Be lenient on grammar**: LLMs are robust to grammatical variations
4. **Be strict on meaning**: Count errors that change intent, actions, or key entities
5. **Possessives and plurals**: Almost never errors unless they change core meaning
6. **Show your semantic reasoning**: Explain WHY something is or isn't an error"""


SEMANTIC_WER_USER_PROMPT_TEMPLATE = """Please calculate the Word Error Rate (WER) for this ASR transcription.

**Reference (ground truth):**
{reference}

**Hypothesis (ASR transcription):**
{hypothesis}

Follow the process: NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT → CALCULATE
Show your work clearly, then call calculate_wer with your verified counts."""


def build_semantic_wer_user_prompt(reference: str, hypothesis: str) -> str:
    """Build the user prompt for semantic WER evaluation."""
    return SEMANTIC_WER_USER_PROMPT_TEMPLATE.format(
        reference=reference,
        hypothesis=hypothesis,
    )
