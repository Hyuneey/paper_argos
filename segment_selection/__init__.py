"""Evidence-aware segment selection utilities for ARGOS experiments."""

from segment_selection.candidate_generator import CandidateSegment, generate_candidates
from segment_selection.selector import SegmentSelector, SelectionResult

__all__ = [
    "CandidateSegment",
    "SegmentSelector",
    "SelectionResult",
    "generate_candidates",
]
