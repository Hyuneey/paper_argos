from typing import Type

import numpy as np

from eval_metrics.base import EvalInterface, MetricInterface
from eval_metrics.metrics import F1Class


class AffiliationF1(EvalInterface):
    """Non-PA range-aware F1 on fixed binary predictions.

    Precision counts how many predicted-positive points fall inside any true
    anomaly event, divided by total predicted-positive points. Recall counts how
    many true anomaly events are hit by at least one prediction, divided by the
    total number of true events. This penalizes flooding an anomaly event with
    many positive points while still rewarding event coverage.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.name = "affiliation f1"
        self.threshold = float(threshold)

    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=float)
        if scores_arr.shape != labels_arr.shape:
            raise ValueError(
                f"scores and labels must have the same shape, got {scores_arr.shape} vs {labels_arr.shape}"
            )

        predictions = (scores_arr >= self.threshold).astype(int)
        gt_labels = (labels_arr > 0.5).astype(int)

        segments = []
        in_segment = False
        start = 0
        for idx, value in enumerate(gt_labels):
            if value and not in_segment:
                in_segment = True
                start = idx
            elif not value and in_segment:
                in_segment = False
                segments.append((start, idx))
        if in_segment:
            segments.append((start, len(gt_labels)))

        predicted_positive = int(predictions.sum())
        if predicted_positive == 0 or not segments:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            hit_segments = 0
            for start, end in segments:
                if int(predictions[start:end].sum()) > 0:
                    hit_segments += 1

            precision = hit_segments / predicted_positive
            recall = hit_segments / len(segments)
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom else 0.0

        return F1Class(
            name=self.name,
            p=float(precision),
            r=float(recall),
            f1=float(f1),
            thres=self.threshold,
        )
