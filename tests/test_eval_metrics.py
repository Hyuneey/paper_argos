import unittest

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


if __name__ == "__main__":
    unittest.main()
