import unittest

import numpy as np

from agent.review_agent import ReviewAgent


class ReviewAgentMetricsTests(unittest.TestCase):
    def test_eval_scores_by_metrics_includes_affiliation_f1(self):
        agent = ReviewAgent.__new__(ReviewAgent)
        scores = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
        labels = np.array([0, 1, 1, 0], dtype=int)

        result = agent.eval_scores_by_metrics(scores, labels, log=False)

        self.assertIn("affiliation f1", result)
        self.assertAlmostEqual(result["affiliation f1"]["precision"], 0.5)
        self.assertAlmostEqual(result["affiliation f1"]["recall"], 1.0)
        self.assertAlmostEqual(result["affiliation f1"]["f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
