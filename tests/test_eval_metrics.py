import unittest

from eval_metrics.affiliation_f1 import AffiliationF1
from eval_metrics.point_f1_fixed import PointF1Fixed


class PointF1FixedTests(unittest.TestCase):
    def test_fixed_threshold_uses_binary_cutoff(self):
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 0, 1, 0]

        metric = PointF1Fixed().calc(scores, labels, None)

        self.assertEqual(metric.thres, 0.5)
        self.assertAlmostEqual(metric.p, 0.5)
        self.assertAlmostEqual(metric.r, 0.5)
        self.assertAlmostEqual(metric.f1, 0.5)
        payload = metric.to_dict()["point-wise fixed f1"]
        self.assertEqual(payload["threshold"], 0.5)


class AffiliationF1Tests(unittest.TestCase):
    def test_affiliation_f1_penalizes_full_segment_flooding(self):
        scores = [0.0, 1.0, 1.0, 0.0]
        labels = [0, 1, 1, 0]

        metric = AffiliationF1().calc(scores, labels, None)
        payload = metric.to_dict()["affiliation f1"]

        self.assertAlmostEqual(payload["precision"], 0.5)
        self.assertAlmostEqual(payload["recall"], 1.0)
        self.assertAlmostEqual(payload["f1"], 2 / 3)
        self.assertNotEqual(payload["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
