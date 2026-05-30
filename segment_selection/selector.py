from dataclasses import dataclass

import pandas as pd

from segment_selection.candidate_generator import CandidateSegment
from segment_selection.utility_scorer import (
    UtilityScore,
    UtilityScorer,
    config_hash,
    load_selector_config,
)


@dataclass(frozen=True)
class SelectionResult:
    selected: CandidateSegment
    selected_score: UtilityScore
    candidate_scores: list[tuple[CandidateSegment, UtilityScore]]
    config_hash: str


class SegmentSelector:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.config = load_selector_config(config_path)
        self.scorer = UtilityScorer(self.config)
        self.config_hash = config_hash(self.config)

    def select(
        self,
        candidates: list[CandidateSegment],
        full_df: pd.DataFrame,
        target_chunk_size: int,
    ) -> SelectionResult:
        if not candidates:
            raise ValueError("No segment candidates were generated")

        scored = [
            (candidate, self.scorer.score(candidate, full_df, target_chunk_size))
            for candidate in candidates
        ]
        eligible = [
            (candidate, score)
            for candidate, score in scored
            if self._is_eligible(candidate, score)
        ]
        pool = eligible or scored
        selected, selected_score = max(
            pool,
            key=lambda item: (
                item[1].total,
                item[1].components.get("normal_contrast", 0.0),
                item[1].components.get("reference_context", 0.0),
                -item[0].length,
                item[0].kind,
            ),
        )
        return SelectionResult(
            selected=selected,
            selected_score=selected_score,
            candidate_scores=scored,
            config_hash=self.config_hash,
        )

    @staticmethod
    def _is_eligible(candidate: CandidateSegment, score: UtilityScore) -> bool:
        components = score.components
        if components.get("anomaly_density", 0.0) >= 1.0:
            return False
        if components.get("normal_context_floor", 0.0) > 0.0:
            return False
        return True
