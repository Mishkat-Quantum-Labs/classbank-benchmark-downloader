# Requirements Document

## Introduction

This feature implements a preprocessing pipeline that converts CHAT (.cha) transcription files from TalkBank's ClassBank dataset into a standardized JSON format. The pipeline scans the `dataset/transcripts/` directory, parses each .cha file using `pylangacq`, transforms the data into a target JSON schema compatible with the project's transcription pipeline, verifies conversion accuracy, and produces a manifest and verification report. The entire pipeline runs as a single Python script (`preprocess_transcripts.py`) and creates output in `dataset/parsed/`.

## Glossary

- **Pipeline**: The `preprocess_transcripts.py` script that orchestrates discovery, parsing, transformation, verification, and reporting stages
- **CHA_File**: A CHAT-format transcription file (`.cha` extension) conforming to TalkBank conventions, located in `dataset/transcripts/`
- **Target_JSON**: The output JSON format containing `metadata`, `segments`, and `full_text` fields as defined by the project's transcription pipeline schema
- **Corpus**: A named subdirectory within `dataset/transcripts/` representing a collection of related classroom recordings (e.g., "Roth", "APT", "TIMSS-Math")
- **Segment**: A single utterance record containing start time, end time, speaker role, and cleaned text
- **Speaker_Code**: A three-letter identifier in CHAT format representing a participant (e.g., "TEA", "STU")
- **Speaker_Role**: A human-readable role label derived from the @Participants header (e.g., "teacher", "student")
- **Manifest**: The `manifest.json` file indexing all successfully converted files with their metadata
- **Verification_Report**: The `verification.json` file containing per-file quality assurance results
- **Pylangacq**: The Python library used to parse CHAT-format files into structured data

## Requirements

### Requirement 1: Corpus Discovery

**User Story:** As a researcher, I want the pipeline to automatically find all .cha files grouped by corpus, so that I can process the entire dataset without manual file listing.

#### Acceptance Criteria

1. WHEN the Pipeline is executed, THE Pipeline SHALL recursively scan `dataset/transcripts/` and identify all files with a `.cha` extension
2. THE Pipeline SHALL group each discovered CHA_File by the name of its immediate parent directory within `dataset/transcripts/`, treating that directory name as the Corpus identifier
3. IF the `dataset/transcripts/` directory does not exist or is not accessible, THEN THE Pipeline SHALL exit with an error message indicating the expected directory path and a non-zero exit code
4. IF no CHA_File entries are found in `dataset/transcripts/`, THEN THE Pipeline SHALL exit with an error message indicating the directory scanned and that zero `.cha` files were found, and a non-zero exit code
5. THE Pipeline SHALL log the count of discovered CHA_File entries per Corpus to standard output

### Requirement 2: CHA File Parsing

**User Story:** As a researcher, I want .cha files parsed using pylangacq, so that I get structured access to utterances, timestamps, and speaker information.

#### Acceptance Criteria

1. WHEN a CHA_File is processed, THE Pipeline SHALL parse the file using Pylangacq with the `strict=False` parameter to handle mor/word misalignment, producing a list of utterances each containing participant code, time marks, and raw utterance text
2. WHEN a CHA_File contains encoding errors, THE Pipeline SHALL open the file with `errors='replace'` to substitute undecodable bytes rather than failing
3. IF Pylangacq raises an exception during parsing of a CHA_File, THEN THE Pipeline SHALL log the error to standard error including the file path and exception message, and continue processing remaining files
4. WHEN a CHA_File is successfully parsed, THE Pipeline SHALL extract from the @Participants header each participant's Speaker_Code, name, and Speaker_Role
5. WHEN a CHA_File contains no utterances after parsing, THE Pipeline SHALL log a warning to standard error with the file path and produce no output file for that CHA_File

### Requirement 3: Speaker Role Mapping

**User Story:** As a researcher, I want speaker codes mapped to human-readable roles, so that the output JSON uses consistent role labels across all corpora.

#### Acceptance Criteria

1. WHEN a CHA_File contains a @Participants header, THE Pipeline SHALL map each Speaker_Code to its corresponding Speaker_Role by extracting the role field (the third positional element in each comma-separated participant entry) from that header, and use that mapped role as the `speaker` value in each output Segment
2. THE Pipeline SHALL convert Speaker_Role values to lowercase in the output using locale-independent case folding (e.g., "Teacher" becomes "teacher", "Other" becomes "other")
3. WHEN a Speaker_Code has no role defined in the @Participants header (role field is empty or absent), THE Pipeline SHALL use the Speaker_Code converted to lowercase as the Speaker_Role
4. WHEN a CHA_File does not contain a @Participants header, THE Pipeline SHALL use each Speaker_Code converted to lowercase as the Speaker_Role for all utterances in that file

### Requirement 4: Timestamp Conversion

**User Story:** As a researcher, I want timestamps in seconds as floats, so that the output is compatible with audio/video processing tools that use seconds.

#### Acceptance Criteria

1. WHEN an utterance has time marks (a two-element tuple of integers from Pylangacq), THE Pipeline SHALL convert the start timestamp from milliseconds to seconds by dividing by 1000.0
2. WHEN an utterance has time marks, THE Pipeline SHALL convert the end timestamp from milliseconds to seconds by dividing by 1000.0
3. WHEN an utterance has no time marks (Pylangacq returns None for the time_marks field), THE Pipeline SHALL set both `start` and `end` to null in the corresponding Segment
4. THE Pipeline SHALL represent converted timestamp values as floating-point numbers rounded to three decimal places (e.g., 3361 ms becomes 3.361, 0 ms becomes 0.0)
5. IF an utterance has a time marks tuple where one element is None, THEN THE Pipeline SHALL set both `start` and `end` to null in the corresponding Segment

### Requirement 5: Text Cleaning

**User Story:** As a researcher, I want utterance text cleaned of CHAT annotation codes, so that I get plain readable text suitable for NLP processing.

#### Acceptance Criteria

1. WHEN transforming utterance text, THE Pipeline SHALL use the participant main tier (the tier keyed by the Speaker_Code, e.g., "TEA" or "STU") as the source text for cleaning
2. WHEN transforming utterance text, THE Pipeline SHALL remove CHAT pause markers matching the patterns `(.)` and `(N.N)` where N is one or more digits (e.g., `(0.3)`, `(0.9)`, `(1.4)`)
3. WHEN transforming utterance text, THE Pipeline SHALL remove CHAT terminator codes matching the patterns `+/.`, `+//?`, `+...`, and `+//.`
4. WHEN transforming utterance text, THE Pipeline SHALL remove inline timestamp markers matching the pattern `\u0015` followed by one or more digits, an underscore, one or more digits, and a closing `\u0015` (e.g., `\u00150_3361\u0015`)
5. WHEN transforming utterance text, THE Pipeline SHALL remove CHAT stress and syllable marker characters (Unicode control characters U+0001 and U+0002) while preserving the text content between them
6. WHEN transforming utterance text, THE Pipeline SHALL remove elongation markers by removing colon characters that appear between alphabetic characters within a word (e.g., `a:nd` becomes `and`, `be:` becomes `be`)
7. WHEN transforming utterance text, THE Pipeline SHALL remove CHAT incomplete-word parenthetical markers by removing parentheses around single characters within words (e.g., `an(d)` becomes `and`)
8. WHEN transforming utterance text, THE Pipeline SHALL collapse multiple whitespace characters into a single space and trim leading and trailing whitespace
9. WHEN cleaning produces an empty string or a string containing only whitespace, THE Pipeline SHALL exclude that Segment from the output

### Requirement 6: Target JSON Schema Output

**User Story:** As a researcher, I want output files that match the project's transcription pipeline schema, so that downstream tools can process CHA-derived and ASR-derived transcripts identically.

#### Acceptance Criteria

1. THE Pipeline SHALL produce Target_JSON files with a top-level `metadata` object containing: `session_id` (string), `duration_seconds` (float), `language` (string), `model` (string with value "cha_parser_v1"), `created_at` (ISO 8601 timestamp string in UTC), `source` (string with relative path to original CHA_File from the project root), and `corpus` (string with Corpus name)
2. THE Pipeline SHALL generate `session_id` by combining the lowercase Corpus name, a hyphen, the lowercase filename without extension, a hyphen, and an 8-character hex string consisting of the first 8 characters of the SHA-256 hash computed from the absolute file path encoded as UTF-8
3. THE Pipeline SHALL compute `duration_seconds` as the maximum end timestamp across all Segments in the file, expressed in seconds rounded to three decimal places; IF no Segments have a non-null end timestamp, THEN THE Pipeline SHALL set `duration_seconds` to 0.0
4. THE Pipeline SHALL extract the `language` value from the first entry of the @Languages header of the CHA_File, mapping three-letter ISO 639-3 codes to two-letter ISO 639-1 codes (e.g., "eng" maps to "en"); IF the @Languages header is missing, THEN THE Pipeline SHALL default `language` to "en"
5. THE Pipeline SHALL produce a `segments` array where each element contains `start` (float or null), `end` (float or null), `speaker` (string containing the mapped Speaker_Role as defined in Requirement 3), and `text` (string containing cleaned utterance text)
6. THE Pipeline SHALL order elements in the `segments` array by ascending `start` timestamp, with null-start Segments placed at the end in their original source order
7. THE Pipeline SHALL produce a `full_text` field containing the concatenation of all Segment text values from the ordered `segments` array, joined by a single space character
8. THE Pipeline SHALL write output files as UTF-8 encoded JSON with 2-space indentation and a trailing newline character

### Requirement 7: Output File Organization

**User Story:** As a researcher, I want output files organized by corpus in a predictable directory structure, so that I can easily locate converted transcripts.

#### Acceptance Criteria

1. THE Pipeline SHALL write each Target_JSON file to `dataset/parsed/{corpus}/{filename}.json` where `{corpus}` matches the source Corpus directory name and `{filename}` matches the source CHA_File name without extension
2. THE Pipeline SHALL create corpus subdirectories within `dataset/parsed/` when they do not already exist
3. THE Pipeline SHALL NOT modify, overwrite, or delete any files outside of `dataset/parsed/`

### Requirement 8: Conversion Verification

**User Story:** As a researcher, I want automated verification of each conversion, so that I can trust the output integrity without manual inspection.

#### Acceptance Criteria

1. WHEN a CHA_File has been converted, THE Pipeline SHALL compare the utterance count in the source CHA_File to the Segment count in the Target_JSON, recording both counts and their absolute difference as a verification check result
2. WHEN a CHA_File has been converted, THE Pipeline SHALL verify that every Speaker_Code present in the source CHA_File appears as a mapped Speaker_Role value in the Target_JSON `segments` array, recording any missing speakers as a verification check result
3. WHEN a CHA_File has been converted, THE Pipeline SHALL compute timestamp coverage as the percentage of Segments that have non-null `start` and non-null `end` values, expressed as a value between 0 and 100 rounded to one decimal place
4. WHEN a CHA_File has been converted, THE Pipeline SHALL compare the total word count of cleaned text in the Target_JSON to the word count of source utterances after applying the same text cleaning rules defined in Requirement 5, where a word is defined as a whitespace-delimited token, recording both counts and the percentage difference as a verification check result
5. WHEN a CHA_File has been converted, THE Pipeline SHALL verify that all Segment timestamps are non-negative and that each Segment's `end` value is greater than or equal to its `start` value, recording any violating Segment indices as a verification check result
6. WHEN a CHA_File has been converted, THE Pipeline SHALL verify that Segment timestamps are monotonically non-decreasing by `start` value when ordered by their position in the `segments` array
7. WHEN a CHA_File has been converted, THE Pipeline SHALL record all verification check results into a per-file checks object containing `utterance_count`, `speaker_mapping`, `timestamp_coverage`, `word_count`, `timestamp_validity`, and `timestamp_order` entries for inclusion in the Verification_Report

### Requirement 9: Verification Result Classification

**User Story:** As a researcher, I want verification results classified by severity, so that I can prioritize which conversions need review.

#### Acceptance Criteria

1. WHEN all of the following conditions are met: utterance count matches exactly, all speakers are present, timestamp coverage is greater than 90%, word count difference is less than or equal to 10%, and no timestamp ordering violations exist, THEN THE Pipeline SHALL classify the file verification result as "pass"
2. WHEN a file does not meet any "fail" condition AND at least one of the following conditions is met: utterance count differs by 1 or 2, or timestamp coverage is between 50% and 90% inclusive, or word count difference is greater than 10% and less than or equal to 25%, THEN THE Pipeline SHALL classify the file verification result as "warn"
3. WHEN at least one of the following conditions is met: utterance count differs by 3 or more, or timestamp coverage is below 50%, or word count difference exceeds 25%, or any timestamp is negative, or any end timestamp is less than its corresponding start timestamp, or start timestamps are not monotonically non-decreasing, THEN THE Pipeline SHALL classify the file verification result as "fail"
4. IF a file meets conditions for multiple severity levels, THEN THE Pipeline SHALL assign the highest severity classification where "fail" is highest, "warn" is middle, and "pass" is lowest
5. IF a verification result is classified as "warn" or "fail", THEN THE Pipeline SHALL include a `reasons` array listing each triggered condition with a message identifying the check name and the observed value compared to the threshold value

### Requirement 10: Verification Report Output

**User Story:** As a researcher, I want a single verification report for the entire batch, so that I can quickly assess overall conversion quality.

#### Acceptance Criteria

1. THE Pipeline SHALL write a `verification.json` file to `dataset/parsed/verification.json`
2. THE Pipeline SHALL structure `verification.json` with a `run_at` field (ISO 8601 timestamp) and a `summary` object containing `total_files` (integer), `passed` (integer), `warnings` (integer), and `failed` (integer) counts
3. THE Pipeline SHALL include a `files` array in `verification.json` where each element contains `source` (string), `output` (string), `corpus` (string), `status` ("pass", "warn", or "fail"), `checks` object with individual check results, and `warnings` or `reasons` array when status is not "pass"

### Requirement 11: Manifest Output

**User Story:** As a researcher, I want a manifest indexing all converted files, so that downstream tools can enumerate available transcripts without scanning the filesystem.

#### Acceptance Criteria

1. THE Pipeline SHALL write a `manifest.json` file to `dataset/parsed/manifest.json`
2. THE Pipeline SHALL structure `manifest.json` with a `created_at` (ISO 8601 timestamp), `total_files` (integer), and `corpora` object mapping each Corpus name to an object containing `files` (array of relative output paths), `total_segments` (integer), and `total_duration_sec` (float)
3. THE Pipeline SHALL include in `manifest.json` only files that were successfully converted (files that failed parsing are excluded)

### Requirement 12: Pipeline Progress Reporting

**User Story:** As a researcher, I want progress feedback while the pipeline runs, so that I know the processing status for large datasets.

#### Acceptance Criteria

1. THE Pipeline SHALL display a progress indicator using the `tqdm` library during file processing, showing the current file count and corpus name
2. THE Pipeline SHALL print a summary upon completion showing total files processed, pass/warn/fail counts, total segments generated, and elapsed time in seconds
3. IF one or more files have a "fail" verification status, THEN THE Pipeline SHALL exit with a non-zero exit code after completing all processing

### Requirement 13: Non-Destructive Operation

**User Story:** As a researcher, I want the pipeline to only create new files, so that existing data and scripts remain untouched.

#### Acceptance Criteria

1. THE Pipeline SHALL NOT modify any existing files in the project directory
2. THE Pipeline SHALL only create new files within the `dataset/parsed/` directory and the `preprocess_transcripts.py` script itself
3. WHEN a Target_JSON file already exists at the output path, THE Pipeline SHALL overwrite it with the new conversion result
