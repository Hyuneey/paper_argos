import unittest

import pandas as pd

from agent.detection_query import build_detection_agent_v3_query
from agent.prompts.detection import build_detection_agent_v3_prompt


class DetectionPromptTests(unittest.TestCase):
    def test_v3_default_prompt_mentions_normal_reference(self):
        prompt = build_detection_agent_v3_prompt(32, "train-LLM-only")
        self.assertIn("NORMAL REFERENCE 0", prompt)
        self.assertIn("reference as context", prompt)

    def test_v3_query_includes_reference_block(self):
        curr_df = pd.DataFrame(
            {
                "value": [1.0, 2.0],
                "label": [1, 1],
                "index": [0, 1],
            }
        )
        ref_df = pd.DataFrame(
            {
                "value": [0.1, 0.2],
                "label": [0, 0],
                "index": [0, 1],
            }
        )

        query = build_detection_agent_v3_query(
            curr_dfs=[curr_df],
            mode="train-LLM-only",
            reference_dfs=[ref_df],
            anomaly_types="# Anomaly Type 1: spike",
        )

        self.assertIn("##### DATA 0", query)
        self.assertIn("##### NORMAL REFERENCE 0", query)
        self.assertIn("##### Anomaly Types BEGIN #####", query)
        self.assertIn("#  # Anomaly Type 1: spike", query)


if __name__ == "__main__":
    unittest.main()
