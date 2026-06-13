"""
CHA-to-JSON Pipeline: Converts CHAT-format (.cha) transcription files
from TalkBank's ClassBank dataset into standardized JSON format.
"""

from pathlib import Path
from typing import Optional
import json
import re
import hashlib
import datetime
import sys
import tempfile
import time

import pylangacq
from tqdm import tqdm

# --- Constants ---

# ISO 639-3 to ISO 639-1 language code mapping (subset relevant to ClassBank corpora)
LANGUAGE_MAP: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
}

# Input base directory containing .cha transcript files organized by corpus
TRANSCRIPTS_DIR: str = "dataset/transcripts/"

# Output base directory for parsed JSON files
PARSED_DIR: str = "dataset/parsed/"


def extract_participants(reader) -> dict[str, str]:
    """Maps speaker codes to lowercase roles from @Participants header.

    Extracts participant entries from the Reader's participants data and maps
    each speaker code to its role field (converted to lowercase). If a
    participant has no role (empty or absent), the speaker code itself is used
    as the role, converted to lowercase.

    Args:
        reader: A pylangacq Reader instance for a single .cha file.

    Returns:
        A dict mapping speaker code (str) to lowercase role (str).
    """
    participants = reader.participants()
    result: dict[str, str] = {}

    if not participants:
        # No @Participants header — no participants to map
        return result

    for participant in participants:
        code = participant.code
        role = participant.role

        if role:
            # Use the role from @Participants header, lowercased
            result[code] = role.casefold()
        else:
            # Fallback: use speaker code converted to lowercase
            result[code] = code.casefold()

    return result


# --- Transformation: Text Cleaning ---


def clean_utterance_text(raw_text: str) -> str:
    """Removes CHAT annotations, returns cleaned plain text.

    Applies regex substitutions in a specific order to strip CHAT-format
    annotation codes from utterance text, producing plain readable text.

    Args:
        raw_text: The raw utterance text from a CHAT transcript tier.

    Returns:
        Cleaned plain text string, or empty string if nothing remains.
    """
    text = raw_text

    # 1. Remove inline timestamps: \u0015<digits>_<digits>\u0015
    text = re.sub(r"\x15\d+_\d+\x15", "", text)

    # 2. Remove stress/syllable markers: U+0001 and U+0002 control characters
    text = re.sub(r"[\x01\x02]", "", text)

    # 3. Remove pause markers: (.) and (N.N) where N is digits
    text = re.sub(r"\(\.\)|\(\d+\.\d+\)", "", text)

    # 4. Remove terminator codes: +/. +//? +... +//.
    text = re.sub(r"\+/\.|\+//\?|\+\.\.\.|\+//\.", "", text)

    # 5. Remove elongation colons:
    #    a) Colon between two alphabetic characters (e.g., a:nd → and)
    text = re.sub(r"([a-zA-Z]):([a-zA-Z])", r"\1\2", text)
    #    b) Trailing colon after alphabetic char (e.g., be: → be)
    text = re.sub(r"([a-zA-Z]):(?=[^a-zA-Z]|$)", r"\1", text)

    # 6. Remove incomplete-word parentheses: single char in parens within word
    text = re.sub(r"(\w)\((\w)\)", r"\1\2", text)

    # 7. Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()

    # Return empty string if result is empty or whitespace-only
    if not text:
        return ""

    return text


# --- Transformation: Timestamp Conversion ---


def convert_time_marks(
    time_marks: Optional[tuple[int, int]],
) -> tuple[Optional[float], Optional[float]]:
    """Converts ms timestamps to seconds rounded to 3 decimal places.

    Takes a tuple of (start_ms, end_ms) integers from pylangacq and converts
    each value to seconds by dividing by 1000.0, rounding to 3 decimal places.

    Args:
        time_marks: A tuple of (start_ms, end_ms) integers, or None.

    Returns:
        A tuple of (start_sec, end_sec) floats rounded to 3 decimal places,
        or (None, None) if input is None or contains None elements.
    """
    if time_marks is None:
        return (None, None)

    start_ms, end_ms = time_marks

    if start_ms is None or end_ms is None:
        return (None, None)

    start_sec = round(start_ms / 1000.0, 3)
    end_sec = round(end_ms / 1000.0, 3)

    return (start_sec, end_sec)


def generate_session_id(corpus: str, filename: str, file_path: Path) -> str:
    """Generates unique session ID from corpus, filename, and path hash.

    Combines the lowercase corpus name, the lowercase filename without its
    extension, and the first 8 characters of the SHA-256 hex digest of the
    absolute file path (UTF-8 encoded).

    Args:
        corpus: The corpus name (e.g., "Roth").
        filename: The filename including extension (e.g., "roth.cha").
        file_path: The Path object for the file.

    Returns:
        A session ID string in the format:
        {lowercase_corpus}-{lowercase_stem}-{first_8_hex_chars}
    """
    corpus_lower = corpus.lower()
    stem_lower = Path(filename).stem.lower()
    abs_path_str = str(file_path.resolve())
    path_hash = hashlib.sha256(abs_path_str.encode("utf-8")).hexdigest()[:8]
    return f"{corpus_lower}-{stem_lower}-{path_hash}"


def transform_file(
    reader, corpus: str, file_path: Path
) -> Optional[dict]:
    """Transforms parsed CHA data into target JSON schema.

    Extracts utterances from the reader, cleans text, converts timestamps,
    maps speakers, sorts segments, and builds the complete output structure.

    Args:
        reader: A pylangacq Reader instance for a single .cha file.
        corpus: The corpus name (parent directory name).
        file_path: The Path object for the source .cha file.

    Returns:
        A dict with 'metadata', 'segments', and 'full_text' keys matching
        the target JSON schema, or None if no valid segments remain.
    """
    # 1. Get speaker code → role mapping
    participants = extract_participants(reader)

    # 2. Get utterances from the reader
    utterances = reader.utterances()

    # 3. Build segments from utterances
    segments: list[dict] = []
    for utterance in utterances:
        # Get main tier text keyed by speaker code
        participant_code = utterance.participant
        if participant_code is None:
            continue
        tiers = utterance.tiers
        if not tiers or participant_code not in tiers:
            continue
        raw_text = tiers[participant_code]

        # Clean the utterance text
        cleaned_text = clean_utterance_text(raw_text)
        if not cleaned_text:
            continue

        # Convert time marks
        start, end = convert_time_marks(utterance.time_marks)

        # Map speaker using participants dict; fallback to code.casefold()
        speaker_role = participants.get(participant_code, participant_code.casefold())

        # Create segment dict
        segments.append({
            "start": start,
            "end": end,
            "speaker": speaker_role,
            "text": cleaned_text,
        })

    # 10. Return None if no segments remain after cleaning
    if not segments:
        return None

    # 4. Sort segments: non-null start ascending, null-start at end (preserving order)
    non_null_segments = [s for s in segments if s["start"] is not None]
    null_segments = [s for s in segments if s["start"] is None]
    non_null_segments.sort(key=lambda s: s["start"])
    segments = non_null_segments + null_segments

    # 5. Compute duration_seconds: max of all non-null end values, rounded to 3 dp
    end_values = [s["end"] for s in segments if s["end"] is not None]
    duration_seconds = round(max(end_values), 3) if end_values else 0.0

    # 6. Build full_text from joined segment texts
    full_text = " ".join(seg["text"] for seg in segments)

    # 7. Extract language from reader.languages()
    languages = reader.languages()
    language_code = "en"  # default
    if languages:
        first_entry = languages[0]
        # Handle both list-of-lists and flat list formats
        if isinstance(first_entry, list):
            first_lang = first_entry[0] if first_entry else None
        else:
            first_lang = first_entry
        if first_lang:
            language_code = LANGUAGE_MAP.get(first_lang, "en")

    # 8. Generate session_id
    session_id = generate_session_id(corpus, file_path.name, file_path)

    # 9. Build metadata
    source = f"dataset/transcripts/{corpus}/{file_path.name}"
    created_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    metadata = {
        "session_id": session_id,
        "duration_seconds": duration_seconds,
        "language": language_code,
        "model": "cha_parser_v1",
        "created_at": created_at,
        "source": source,
        "corpus": corpus,
    }

    # 11. Return the complete output structure
    return {
        "metadata": metadata,
        "segments": segments,
        "full_text": full_text,
    }


# --- Verification ---


def verify_conversion(
    reader, output: dict, source_path: Path
) -> dict:
    """Runs all verification checks, returns checks dict with status.

    Compares the source CHA file data (via the reader) against the transformed
    output JSON to verify conversion accuracy across multiple dimensions.

    Args:
        reader: A pylangacq Reader instance for the source .cha file.
        output: The dict returned by transform_file (has keys: metadata, segments, full_text).
        source_path: The Path to the source .cha file.

    Returns:
        A dict containing source/output paths, corpus, status, checks, and reasons.
    """
    segments = output["segments"]
    corpus = output["metadata"]["corpus"]

    # Derive relative paths
    source_rel = str(source_path.as_posix())
    # Make relative if it contains dataset/transcripts/
    if "dataset/transcripts/" in source_rel:
        source_rel = "dataset/transcripts/" + source_rel.split("dataset/transcripts/")[-1]
    else:
        source_rel = str(source_path)

    output_rel = f"dataset/parsed/{corpus}/{source_path.stem}.json"

    # --- 1. Utterance count (Req 8.1) ---
    source_utterances = list(reader.utterances())
    # Count only utterances that have a valid participant tier and produce
    # non-empty text after cleaning (matching what transform_file outputs)
    source_count = 0
    for utterance in source_utterances:
        if utterance.participant is None:
            continue
        if utterance.tiers is None or utterance.participant not in utterance.tiers:
            continue
        cleaned = clean_utterance_text(utterance.tiers[utterance.participant])
        if cleaned:
            source_count += 1
    output_count = len(segments)
    utterance_diff = abs(source_count - output_count)

    # --- 2. Speaker mapping completeness (Req 8.2) ---
    # Get all unique speaker codes from source utterances
    source_speakers = set()
    for utterance in source_utterances:
        if utterance.participant is not None:
            source_speakers.add(utterance.participant)

    # Get participant mapping (code → role)
    participants_map = extract_participants(reader)

    # Map source speakers to their expected roles
    expected_roles = sorted(set(
        participants_map.get(code, code.casefold()) for code in source_speakers
    ))

    # Get all unique speaker roles from output segments
    found_roles = sorted(set(seg["speaker"] for seg in segments))

    # Find missing roles (expected but not found in output)
    missing_roles = sorted(set(expected_roles) - set(found_roles))

    # --- 3. Timestamp coverage (Req 8.3) ---
    total_segments = len(segments)
    if total_segments > 0:
        segments_with_timestamps = sum(
            1 for seg in segments
            if seg["start"] is not None and seg["end"] is not None
        )
        timestamp_coverage = round(
            (segments_with_timestamps / total_segments) * 100, 1
        )
    else:
        timestamp_coverage = 0.0

    # --- 4. Word count comparison (Req 8.4) ---
    # Count words in output segments
    output_word_count = sum(
        len(seg["text"].split()) for seg in segments
    )

    # Count words in source by cleaning each utterance text
    source_word_count = 0
    for utterance in source_utterances:
        participant_code = utterance.participant
        if participant_code is None:
            continue
        tiers = utterance.tiers
        if not tiers or participant_code not in tiers:
            continue
        raw_text = tiers[participant_code]
        cleaned = clean_utterance_text(raw_text)
        if cleaned:
            source_word_count += len(cleaned.split())

    # Compute percentage difference
    if source_word_count > 0:
        word_diff_percent = round(
            abs(output_word_count - source_word_count) / source_word_count * 100, 1
        )
    else:
        word_diff_percent = 0.0

    # --- 5. Timestamp validity (Req 8.5) ---
    validity_violations: list[int] = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        if start is not None and end is not None:
            if start < 0 or end < 0 or end < start:
                validity_violations.append(i)

    # --- 6. Timestamp monotonicity (Req 8.6) ---
    non_null_starts = [
        seg["start"] for seg in segments if seg["start"] is not None
    ]
    monotonic = True
    for i in range(1, len(non_null_starts)):
        if non_null_starts[i] < non_null_starts[i - 1]:
            monotonic = False
            break

    # --- Build result dict ---
    return {
        "source": source_rel,
        "output": output_rel,
        "corpus": corpus,
        "status": "",  # to be filled by classify_verification
        "checks": {
            "utterance_count": {
                "source": source_count,
                "output": output_count,
                "diff": utterance_diff,
            },
            "speaker_mapping": {
                "expected": expected_roles,
                "found": found_roles,
                "missing": missing_roles,
            },
            "timestamp_coverage": {
                "percentage": timestamp_coverage,
            },
            "word_count": {
                "source": source_word_count,
                "output": output_word_count,
                "diff_percent": word_diff_percent,
            },
            "timestamp_validity": {
                "violations": validity_violations,
            },
            "timestamp_order": {
                "monotonic": monotonic,
            },
        },
        "reasons": [],  # to be populated by classify_verification later
    }


def classify_verification(checks: dict) -> tuple[str, list[str]]:
    """Returns (status, reasons) based on check results.

    Applies pass/warn/fail thresholds to the verification checks dict and
    returns the highest-severity classification along with descriptive reasons
    for any triggered conditions.

    Priority: fail > warn > pass.

    Args:
        checks: The "checks" sub-dict from verify_conversion containing
            utterance_count, speaker_mapping, timestamp_coverage,
            word_count, timestamp_validity, and timestamp_order entries.

    Returns:
        A tuple of (status, reasons) where status is "pass", "warn", or "fail",
        and reasons is a list of descriptive messages (empty for "pass").
    """
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    # --- Extract check values ---
    utterance_diff = checks["utterance_count"]["diff"]
    missing_speakers = checks["speaker_mapping"]["missing"]
    timestamp_coverage = checks["timestamp_coverage"]["percentage"]
    word_diff_percent = checks["word_count"]["diff_percent"]
    validity_violations = checks["timestamp_validity"]["violations"]
    monotonic = checks["timestamp_order"]["monotonic"]

    # --- FAIL conditions (Req 9.3) ---
    if utterance_diff >= 3:
        fail_reasons.append(
            f"utterance_count: diff={utterance_diff} (threshold: 3)"
        )

    # Timestamp coverage only triggers a warn, never a fail.
    # Low/zero coverage indicates missing time marks in the source data,
    # not a conversion quality issue.
    if timestamp_coverage <= 90:
        warn_reasons.append(
            f"timestamp_coverage: {timestamp_coverage}% (threshold: <=90%)"
        )

    if word_diff_percent > 25:
        fail_reasons.append(
            f"word_count: diff={word_diff_percent}% (threshold: >25%)"
        )

    if validity_violations:
        fail_reasons.append(
            f"timestamp_validity: violations at indices {validity_violations}"
        )

    if not monotonic:
        fail_reasons.append("timestamp_order: not monotonic")

    # --- WARN conditions (Req 9.2) ---
    if 1 <= utterance_diff <= 2:
        warn_reasons.append(
            f"utterance_count: diff={utterance_diff} (threshold: 1-2)"
        )

    if 10 < word_diff_percent <= 25:
        warn_reasons.append(
            f"word_count: diff={word_diff_percent}% (threshold: >10%)"
        )

    # --- Speaker mapping missing (contributes to pass check, Req 9.1) ---
    if missing_speakers:
        warn_reasons.append(
            f"speaker_mapping: missing roles {missing_speakers}"
        )

    # --- Classify with priority: fail > warn > pass (Req 9.4) ---
    if fail_reasons:
        return ("fail", fail_reasons)
    elif warn_reasons:
        return ("warn", warn_reasons)
    else:
        return ("pass", [])


# --- Parsing ---


def parse_cha_file(file_path: Path):
    """Parses a .cha file, returns Reader or None on failure.

    Opens the file with errors='replace' to handle encoding issues,
    writes sanitized content to a temporary file, then passes it
    to pylangacq for parsing with strict=False.
    """
    try:
        # Read file with errors='replace' for encoding resilience (Req 2.2)
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Write sanitized content to a temporary file for pylangacq to read
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cha", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Parse with strict=False to handle mor/word misalignment (Req 2.1)
            reader = pylangacq.read_chat(tmp_path, strict=False)
            return reader
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        # Log error to stderr and return None (Req 2.3)
        print(
            f"[ERROR] Failed to parse: {file_path} - {e}",
            file=sys.stderr,
        )
        return None


# --- Discovery ---


# --- Reporting ---


def generate_manifest(results: list[dict]) -> dict:
    """Builds manifest.json structure from successful conversions.

    Filters results to only include successfully converted files, groups them
    by corpus name, and aggregates segment counts and durations per corpus.

    Args:
        results: A list of result dicts, one per processed file. Each dict has:
            - corpus (str): The corpus name
            - source_path (Path): Path to the source .cha file
            - output_path (Path or None): Path to the output .json file
            - success (bool): Whether conversion succeeded
            - segment_count (int): Number of segments produced
            - duration (float): Duration in seconds
            - verification (dict or None): Verification results

    Returns:
        A dict with manifest structure containing:
            - created_at: ISO 8601 UTC timestamp
            - total_files: Count of successful files only
            - corpora: Mapping of corpus name to files/segments/duration
    """
    # Filter to only successful conversions (Req 11.3)
    successful = [r for r in results if r.get("success")]

    # Group by corpus
    corpora: dict[str, dict] = {}
    for result in successful:
        corpus_name = result["corpus"]
        if corpus_name not in corpora:
            corpora[corpus_name] = {
                "files": [],
                "total_segments": 0,
                "total_duration_sec": 0.0,
            }

        # Add relative output path
        output_path = result["output_path"]
        if output_path is not None:
            # Convert to forward-slash relative path
            rel_path = Path(output_path).as_posix()
            # Ensure path is relative (starts with dataset/parsed/...)
            if "dataset/parsed/" in rel_path:
                rel_path = "dataset/parsed/" + rel_path.split("dataset/parsed/")[-1]
            corpora[corpus_name]["files"].append(rel_path)

        # Aggregate segment count and duration
        corpora[corpus_name]["total_segments"] += result["segment_count"]
        corpora[corpus_name]["total_duration_sec"] += result["duration"]

    # Build manifest
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "total_files": len(successful),
        "corpora": corpora,
    }

    return manifest


def generate_verification_report(results: list[dict]) -> dict:
    """Builds verification.json structure from all results.

    Creates a verification report containing a run timestamp, summary counts
    (total, passed, warnings, failed), and a files array with per-file
    verification details. Only includes files that have a non-None verification
    dict (files that failed to parse are excluded).

    Args:
        results: A list of result dicts, one per processed file. Each dict has:
            - corpus (str): The corpus name
            - source_path (Path): Path to the source .cha file
            - output_path (Path or None): Path to the output .json file
            - success (bool): Whether conversion succeeded
            - segment_count (int): Number of segments produced
            - duration (float): Duration in seconds
            - verification (dict or None): The verification result dict with
              source, output, corpus, status, checks, and reasons fields.

    Returns:
        A dict with verification report structure containing:
            - run_at: ISO 8601 UTC timestamp
            - summary: Dict with total_files, passed, warnings, failed counts
            - files: Array of per-file verification results
    """
    # Filter to only results with a non-None verification dict (Req 10.3)
    verified_results = [r for r in results if r.get("verification") is not None]

    # Count statuses
    passed = 0
    warnings = 0
    failed = 0

    files_array: list[dict] = []

    for result in verified_results:
        verification = result["verification"]
        status = verification.get("status", "")

        if status == "pass":
            passed += 1
        elif status == "warn":
            warnings += 1
        elif status == "fail":
            failed += 1

        # Build file entry from the verification dict
        file_entry: dict = {
            "source": verification["source"],
            "output": verification["output"],
            "corpus": verification["corpus"],
            "status": verification["status"],
            "checks": verification["checks"],
        }

        # Include reasons when status is not "pass" (Req 10.3)
        if status != "pass":
            file_entry["reasons"] = verification.get("reasons", [])

        files_array.append(file_entry)

    total_files = len(verified_results)

    # Build verification report
    report = {
        "run_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "summary": {
            "total_files": total_files,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
        },
        "files": files_array,
    }

    return report


def discover_cha_files(base_dir: str) -> list[tuple[str, Path]]:
    """Returns list of (corpus_name, file_path) tuples.

    Recursively scans base_dir for .cha files, grouping them by their
    immediate parent directory name as the corpus identifier.

    Exits with code 1 if the directory is missing or no .cha files are found.
    Prints per-corpus file counts to stdout.
    """
    base_path = Path(base_dir)

    # Check if directory exists and is accessible
    if not base_path.exists() or not base_path.is_dir():
        print(
            f"[ERROR] Directory not found: {base_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Recursively find all .cha files
    cha_files: list[tuple[str, Path]] = []
    for file_path in sorted(base_path.rglob("*.cha")):
        corpus_name = file_path.parent.name
        cha_files.append((corpus_name, file_path))

    # Exit if no .cha files found
    if not cha_files:
        print(
            f"[ERROR] No .cha files found in: {base_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Count files per corpus
    corpus_counts: dict[str, int] = {}
    for corpus_name, _ in cha_files:
        corpus_counts[corpus_name] = corpus_counts.get(corpus_name, 0) + 1

    # Print per-corpus counts
    for corpus, count in sorted(corpus_counts.items()):
        print(f"[INFO]  {corpus}: {count} file(s)")

    total_files = len(cha_files)
    total_corpora = len(corpus_counts)
    print(f"[INFO]  Discovered {total_files} .cha files across {total_corpora} corpora")

    return cha_files


# --- Main Orchestration ---


def main() -> int:
    """Entry point. Returns exit code."""
    start_time = time.time()

    # 1. Discovery stage
    cha_files = discover_cha_files(TRANSCRIPTS_DIR)

    # 2. Initialize results list
    results: list[dict] = []

    # 3. Processing loop with tqdm progress bar
    for corpus, file_path in tqdm(cha_files, desc="Processing", unit="file"):
        # 3a. Parse the CHA file
        reader = parse_cha_file(file_path)
        if reader is None:
            # Parse failed — append failed result and continue
            results.append({
                "corpus": corpus,
                "source_path": file_path,
                "output_path": None,
                "success": False,
                "segment_count": 0,
                "duration": 0.0,
                "verification": None,
            })
            continue

        # 3b. Check if reader has utterances
        utterances = list(reader.utterances())
        if not utterances:
            print(
                f"[WARN]  No utterances found: {file_path}",
                file=sys.stderr,
            )
            results.append({
                "corpus": corpus,
                "source_path": file_path,
                "output_path": None,
                "success": False,
                "segment_count": 0,
                "duration": 0.0,
                "verification": None,
            })
            continue

        # 3c. Transform file to JSON structure
        output = transform_file(reader, corpus, file_path)
        if output is None:
            # No valid segments after cleaning
            print(
                f"[WARN]  No valid segments after cleaning: {file_path}",
                file=sys.stderr,
            )
            results.append({
                "corpus": corpus,
                "source_path": file_path,
                "output_path": None,
                "success": False,
                "segment_count": 0,
                "duration": 0.0,
                "verification": None,
            })
            continue

        # 3d. Write output JSON
        output_dir = Path(PARSED_DIR) / corpus
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_path.stem}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            f.write("\n")  # trailing newline

        # 3e. Verify conversion
        verification = verify_conversion(reader, output, file_path)

        # 3f. Classify verification result
        status, reasons = classify_verification(verification["checks"])

        # 3g. Set status and reasons on verification dict
        verification["status"] = status
        verification["reasons"] = reasons

        # 3h. Append success result
        results.append({
            "corpus": corpus,
            "source_path": file_path,
            "output_path": output_path,
            "success": True,
            "segment_count": len(output["segments"]),
            "duration": output["metadata"]["duration_seconds"],
            "verification": verification,
        })

    # 4. Generate and write manifest
    manifest = generate_manifest(results)
    parsed_dir = Path(PARSED_DIR)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    with open(parsed_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # 5. Generate and write verification report
    verification_report = generate_verification_report(results)

    with open(parsed_dir / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # 6. Print summary
    elapsed = time.time() - start_time
    total_files = len(results)
    pass_count = sum(
        1 for r in results
        if r.get("verification") and r["verification"].get("status") == "pass"
    )
    warn_count = sum(
        1 for r in results
        if r.get("verification") and r["verification"].get("status") == "warn"
    )
    fail_count = sum(
        1 for r in results
        if r.get("verification") and r["verification"].get("status") == "fail"
    )
    # Count files without verification (parse/transform failures) as fail
    fail_count += sum(1 for r in results if not r.get("success"))
    total_segments = sum(r["segment_count"] for r in results)

    print("\n--- Summary ---")
    print(f"Total files processed: {total_files}")
    print(f"Pass: {pass_count} | Warn: {warn_count} | Fail: {fail_count}")
    print(f"Total segments: {total_segments}")
    print(f"Elapsed time: {elapsed:.1f}s")

    # 7. Exit with code 1 if any file has "fail" status, else 0
    has_fail = any(
        r.get("verification") and r["verification"].get("status") == "fail"
        for r in results
    )
    # Also consider files that failed to parse/transform as failures
    has_fail = has_fail or any(not r.get("success") for r in results)

    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
