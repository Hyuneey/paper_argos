from typing import Type

import numpy as np

from eval_metrics.base import EvalInterface, MetricInterface
from eval_metrics.metrics import F1Class


class PointF1Fixed(EvalInterface):
    """Point-wise F1 at a fixed threshold.

    This is the measurement-safe version of point F1: no threshold search, no
    oracle-on-test behavior. Predictions are obtained by thresholding scores at
    the fixed cutoff (default 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.name = "point-wise fixed f1"
        self.threshold = float(threshold)

    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=float)
        if scores_arr.shape != labels_arr.shape:
            raise ValueError(
                f"scores and labels must have the same shape, got {scores_arr.shape} vs {labels_arr.shape}"
            )

        predictions = (scores_arr >= self.threshold).astype(int)
        labels_bin = (labels_arr > 0.5).astype(int)

        tp = int(np.sum((predictions == 1) & (labels_bin == 1)))
        fp = int(np.sum((predictions == 1) & (labels_bin == 0)))
        fn = int(np.sum((predictions == 0) & (labels_bin == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0

        return F1Class(
            name=self.name,
            p=float(precision),
            r=float(recall),
            f1=float(f1),
            thres=self.threshold,
        )
