# Implementation Plan: CHA-to-JSON Pipeline

## Overview

Implement a single Python script (`preprocess_transcripts.py`) that converts CHAT-format (.cha) transcription files into standardized JSON. The script progresses through five stages: discovery, parsing, transformation, verification, and reporting. All output goes into `dataset/parsed/`. A comprehensive test suite validates each stage using property-based testing with Hypothesis.

## Tasks

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create `preprocess_transcripts.py` with imports and constants
    - Create the script file with all necessary imports: `pathlib`, `json`, `re`, `hashlib`, `datetime`, `sys`, `typing`, `pylangacq`, `tqdm`
    - Define the `LANGUAGE_MAP` constant mapping ISO 639-3 to ISO 639-1 codes
    - Define output base directory constant `PARSED_DIR = "dataset/parsed/"`
    - Define input base directory constant `TRANSCRIPTS_DIR = "dataset/transcripts/"`
    - _Requirements: 1.1, 6.1, 6.4_

  - [x] 1.2 Set up test infrastructure
    - Create `tests/` directory with `__init__.py`
    - Create `tests/conftest.py` with shared fixtures (sample utterance data, mock readers)
    - Ensure `pytest` and `hypothesis` are available (add to requirements if needed)
    - _Requirements: Design Testing Strategy_

- [x] 2. Implement discovery stage
  - [x] 2.1 Implement `discover_cha_files` function
    - Recursively scan `dataset/transcripts/` using `Path.rglob("*.cha")`
    - Group files by immediate parent directory name as corpus identifier
    - Exit with code 1 and error message to stderr if directory is missing or no .cha files found
    - Print per-corpus file counts to stdout
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test for corpus grouping (Property 1)
    - **Property 1: Corpus grouping matches parent directory**
    - For any file path within a transcripts directory, the corpus identifier equals the immediate parent directory name
    - **Validates: Requirements 1.2**

  - [ ]* 2.3 Write unit tests for discovery error handling
    - Test missing directory raises SystemExit with code 1
    - Test empty directory (no .cha files) raises SystemExit with code 1
    - Test stdout output includes per-corpus counts
    - _Requirements: 1.3, 1.4, 1.5_

- [x] 3. Implement parsing stage
  - [x] 3.1 Implement `parse_cha_file` function
    - Open file with `errors='replace'` for encoding resilience
    - Call `pylangacq.read_chat(file_path, strict=False)`
    - Return Reader on success, None on exception (log error to stderr)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Implement `extract_participants` function
    - Extract participant entries from Reader's participants data
    - Map speaker codes to their role field from @Participants header
    - Return dict mapping code → lowercase role
    - _Requirements: 2.4, 3.1, 3.2, 3.3, 3.4_

  - [ ]* 3.3 Write property test for speaker role resolution (Property 2)
    - **Property 2: Speaker role resolution**
    - For any participant with a defined role, output is lowercase role; for any without a role, output is lowercase speaker code
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

  - [ ]* 3.4 Write unit tests for parsing edge cases
    - Test exception handling logs error and returns None
    - Test file with no utterances logs warning and produces no output
    - Test encoding error recovery with `errors='replace'`
    - _Requirements: 2.2, 2.3, 2.5_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement transformation stage - text cleaning
  - [x] 5.1 Implement `clean_utterance_text` function
    - Apply regex substitutions in order: inline timestamps, stress/syllable markers, pause markers, terminator codes, elongation colons, incomplete-word parentheses, whitespace collapse
    - Return cleaned string or empty string
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [ ]* 5.2 Write property test for text cleaning idempotency (Property 4)
    - **Property 4: Text cleaning is idempotent**
    - For any string, `clean(clean(x)) == clean(x)`
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**

  - [ ]* 5.3 Write property test for annotation removal (Property 5)
    - **Property 5: Cleaned text contains no CHAT annotation patterns**
    - For any input, cleaned output contains no pause markers, terminator codes, inline timestamps, stress/syllable characters, or elongation colons between alphabetic chars
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.6**

- [x] 6. Implement transformation stage - core transform
  - [x] 6.1 Implement `convert_time_marks` function
    - Convert ms tuple to seconds tuple (divide by 1000.0, round to 3 decimal places)
    - Return (None, None) when input is None or contains None elements
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 6.2 Implement `generate_session_id` function
    - Combine lowercase corpus, hyphen, lowercase filename (no extension), hyphen, first 8 chars of SHA-256 hex of absolute path
    - _Requirements: 6.2_

  - [x] 6.3 Implement `transform_file` function
    - Extract utterances from reader, clean text, convert timestamps, map speakers
    - Sort segments by start timestamp with null-starts at end
    - Compute `duration_seconds` as max end timestamp (or 0.0)
    - Build `full_text` from joined segment texts
    - Construct complete metadata object with session_id, language, model, created_at, source, corpus
    - Return None if no valid segments remain
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 6.4 Write property test for timestamp conversion (Property 3)
    - **Property 3: Timestamp conversion preserves value**
    - For any non-negative integer ms, converting to seconds equals ms / 1000.0 rounded to 3 decimal places; None inputs yield null outputs
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

  - [ ]* 6.5 Write property test for segment ordering (Property 6)
    - **Property 6: Segments are ordered by start timestamp with nulls last**
    - For any output segments array, non-null starts are non-decreasing and null-start segments appear after all non-null-start segments
    - **Validates: Requirements 6.6**

  - [ ]* 6.6 Write property test for full_text construction (Property 7)
    - **Property 7: Full text equals space-joined segment texts**
    - For any valid output, `full_text` equals `" ".join(seg["text"] for seg in segments)`
    - **Validates: Requirements 6.7**

  - [ ]* 6.7 Write property test for session ID (Property 8)
    - **Property 8: Session ID is deterministic and correctly formatted**
    - For any corpus, filename, and path, session_id is always the same value and matches `{corpus_lower}-{filename_lower}-{hash8}`
    - **Validates: Requirements 6.2**

  - [ ]* 6.8 Write property test for duration calculation (Property 9)
    - **Property 9: Duration equals maximum end timestamp**
    - For any set of segments, `duration_seconds` equals max non-null end rounded to 3 decimal places, or 0.0 if none
    - **Validates: Requirements 6.3**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement verification stage
  - [x] 8.1 Implement `verify_conversion` function
    - Compare utterance count (source vs output segment count)
    - Check speaker mapping completeness (all source speakers present in output)
    - Compute timestamp coverage percentage
    - Compare word counts with percentage difference
    - Check timestamp validity (non-negative, end >= start)
    - Check timestamp monotonicity (non-decreasing start values)
    - Return checks dict with all results
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x] 8.2 Implement `classify_verification` function
    - Apply pass/warn/fail thresholds per design specification
    - Return (status, reasons) tuple
    - "fail" takes precedence over "warn", "warn" over "pass"
    - Include descriptive reason messages with check name and observed values
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 8.3 Write property test for verification classification (Property 10)
    - **Property 10: Verification classification is consistent with check values**
    - For any check results, classification assigns correct severity; reasons array is non-empty when status != "pass"
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

  - [ ]* 8.4 Write property test for timestamp validity detection (Property 11)
    - **Property 11: Timestamp validity detection**
    - For any segment with non-null timestamps, violation reported iff start < 0, end < 0, or end < start
    - **Validates: Requirements 8.5**

  - [ ]* 8.5 Write property test for timestamp monotonicity detection (Property 12)
    - **Property 12: Timestamp monotonicity detection**
    - For any sequence of segments, monotonicity violation reported iff consecutive non-null starts decrease
    - **Validates: Requirements 8.6**

- [x] 9. Implement reporting stage
  - [x] 9.1 Implement `generate_manifest` function
    - Build manifest structure with `created_at`, `total_files`, and `corpora` mapping
    - Include only successfully converted files
    - Aggregate `total_segments` and `total_duration_sec` per corpus
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 9.2 Implement `generate_verification_report` function
    - Build verification report with `run_at`, `summary` counts, and `files` array
    - Include all processed files (success and failure)
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ]* 9.3 Write property test for manifest filtering (Property 13)
    - **Property 13: Manifest includes only successfully converted files**
    - For any mix of success/failure results, manifest count equals success count and contains no failed files
    - **Validates: Requirements 11.3**

- [x] 10. Implement main orchestration and output writing
  - [x] 10.1 Implement `main` function and file output logic
    - Orchestrate all stages: discover → parse → transform → verify → report
    - Write each Target_JSON to `dataset/parsed/{corpus}/{filename}.json` with 2-space indent and trailing newline
    - Create corpus subdirectories as needed
    - Write `manifest.json` and `verification.json` to `dataset/parsed/`
    - Display tqdm progress bar during processing
    - Print completion summary (total files, pass/warn/fail, total segments, elapsed time)
    - Exit with code 1 if any file has "fail" status, else 0
    - _Requirements: 7.1, 7.2, 7.3, 6.8, 12.1, 12.2, 12.3, 13.1, 13.2, 13.3_

  - [ ]* 10.2 Write integration tests
    - Test full pipeline with real `dataset/transcripts/Roth/roth.cha` file
    - Verify output JSON structure matches schema
    - Verify manifest and verification report are generated
    - Verify tqdm is invoked for progress display
    - _Requirements: 6.1, 10.1, 11.1, 12.1_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation language is Python (as specified in the design)
- All logic resides in a single script `preprocess_transcripts.py`
- Test files go in `tests/` directory using pytest + hypothesis

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1", "3.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.3", "3.4", "5.1"] },
    { "id": 3, "tasks": ["5.2", "5.3", "6.1", "6.2"] },
    { "id": 4, "tasks": ["6.3", "6.4", "6.7"] },
    { "id": 5, "tasks": ["6.5", "6.6", "6.8", "8.1"] },
    { "id": 6, "tasks": ["8.2", "8.3", "8.4", "8.5"] },
    { "id": 7, "tasks": ["9.1", "9.2"] },
    { "id": 8, "tasks": ["9.3", "10.1"] },
    { "id": 9, "tasks": ["10.2"] }
  ]
}
```
