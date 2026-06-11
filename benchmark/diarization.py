"""Diarization Error Rate — using pyannote.metrics (industry standard).

Uses the same tool as NIST evaluations, AssemblyAI SDK, and academic papers.
Implements Hungarian algorithm for optimal speaker mapping with 0.25s collar.
"""

import logging
from typing import Optional

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

logger = logging.getLogger("benchmark")

# Industry standard: 0.25s collar around segment boundaries
# This forgives small timing differences at speaker turns
COLLAR = 0.25


def compute_diarization_error_rate(
    reference_segments: list[dict],
    hypothesis_segments: list[dict],
    collar: float = COLLAR,
) -> Optional[float]:
    """Compute DER using pyannote.metrics (industry standard).

    Uses Hungarian algorithm for optimal speaker mapping.
    Applies collar to forgive boundary timing differences.

    Args:
        reference_segments: Ground truth segments with 'start', 'end', 'speaker'.
        hypothesis_segments: Engine output segments with 'start', 'end', 'speaker'.
        collar: Duration (seconds) of forgiveness around boundaries. Default 0.25s.

    Returns:
        DER as float (0.0 = perfect), or None if computation not possible.
    """
    if not reference_segments or not hypothesis_segments:
        return None

    # Validate required fields and values
    for seg in reference_segments:
        if not all(k in seg for k in ("start", "end", "speaker")):
            return None
    for seg in hypothesis_segments:
        if not all(k in seg for k in ("start", "end", "speaker")):
            return None

    # Build pyannote Annotations, skipping segments with invalid timestamps
    reference = Annotation()
    for seg in reference_segments:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (TypeError, ValueError):
            continue
        if seg["speaker"] and end > start:
            reference[Segment(start, end)] = seg["speaker"]

    hypothesis = Annotation()
    for seg in hypothesis_segments:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (TypeError, ValueError):
            continue
        if seg["speaker"] and end > start:
            hypothesis[Segment(start, end)] = seg["speaker"]

    if not reference or not hypothesis:
        return None

    # Compute DER with Hungarian algorithm + collar
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    der = metric(reference, hypothesis)

    return round(float(der), 4)
