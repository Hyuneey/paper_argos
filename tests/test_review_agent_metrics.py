import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_rule_evidence_consistency import (
    _evaluate_run_directory,
    _paired_delta_rows,
    _summarize_by_condition,
)


class ReviewAgentMetricsTests(unittest.TestCase):
    def test_evaluate_run_directory_computes_heldout_and_local_support_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "series.csv"
            df = pd.DataFrame(
                {
                    "value": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                    "label": [1, 1, 1, 0, 0, 0],
                    "index": list(range(6)),
                }
            )
            df.to_csv(dataset_path, index=False)

            rule_path = tmp / "rule.py"
            rule_path.write_text(
                """
import numpy as np


def inference(sample: np.ndarray) -> np.ndarray:
    values = np.asarray(sample[:, 0], dtype=float)
    return (values > 4.0).astype(int)
""".strip(),
                encoding="utf-8",
            )

            trace_path = tmp / "selection_trace_iter_0_call_0.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "provenance": {"selected_candidate_type": "event_bounded_short"},
                        "selected_segment": {"start_pos": 0, "end_pos": 2},
                        "reference_segments": {"normal_reference": {"start_pos": 3, "end_pos": 5}},
                    }
                ),
                encoding="utf-8",
            )

            pool_path = tmp / "held_out_window_pool.json"
            pool_path.write_text(
                json.dumps(
                    {
                        "pool": [
                            {
                                "split": "val",
                                "chunk_id": 0,
                                "start_pos": 0,
                                "end_pos": 2,
                                "total_points": 3,
                                "anomaly_point_count": 1,
                                "anomaly_event_count": 1,
                            },
                            {
                                "split": "val",
                                "chunk_id": 1,
                                "start_pos": 3,
                                "end_pos": 5,
                                "total_points": 3,
                                "anomaly_point_count": 0,
                                "anomaly_event_count": 0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            run_dir = tmp / "run_01"
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset": "series",
                        "dataset_path": str(dataset_path),
                        "chunk_size": 3,
                        "repeat_id": 1,
                        "top_k": 5,
                        "max_iter": 1,
                        "seed": 8,
                        "llm_provider": "chatgpt-oauth",
                        "resolved_llm_engine": "gpt-5.4-mini",
                        "temperature": 0.0,
                        "segment_selection_mode": "evidence",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "stats.json").write_text(
                json.dumps(
                    {
                        "best_rule_paths": [str(rule_path)],
                        "held_out_window_pool_path": str(pool_path),
                        "held_out_window_pool_count": 2,
                        "selection_trace_paths": [str(trace_path)],
                    }
                ),
                encoding="utf-8",
            )

            result = _evaluate_run_directory(run_dir)

            if result is None:
                self.fail("Expected evaluation result")
            row = result.row
            self.assertEqual(row["heldout_anomaly_support_rate"], 1.0)
            self.assertEqual(row["heldout_normal_violation_rate"], 0.0)
            self.assertEqual(row["heldout_support_gap"], 1.0)
            self.assertEqual(row["local_evidence_support_rate"], 1.0)
            self.assertEqual(row["local_normal_reference_violation_rate"], 0.0)
            self.assertEqual(row["local_support_gap"], 1.0)

    def test_condition_summary_and_pair_delta_include_support_gap(self):
        rows = [
            {
                "dataset": "series",
                "series": "series",
                "condition": "fixed",
                "chunk_size": 3,
                "repeat_id": 1,
                "top_k": 5,
                "max_iter": 1,
                "seed": 8,
                "heldout_anomaly_support_rate": 1.0,
                "heldout_normal_violation_rate": 0.0,
                "heldout_support_gap": 1.0,
                "local_evidence_support_rate": 1.0,
                "local_normal_reference_violation_rate": 0.0,
                "local_support_gap": 1.0,
                "held_out_anomaly_window_count": 1,
                "held_out_normal_window_count": 1,
                "held_out_total_window_count": 2,
            },
            {
                "dataset": "series",
                "series": "series",
                "condition": "evidence",
                "chunk_size": 3,
                "repeat_id": 1,
                "top_k": 5,
                "max_iter": 1,
                "seed": 8,
                "heldout_anomaly_support_rate": 0.0,
                "heldout_normal_violation_rate": 0.0,
                "heldout_support_gap": 0.0,
                "local_evidence_support_rate": 0.0,
                "local_normal_reference_violation_rate": 0.0,
                "local_support_gap": 0.0,
                "held_out_anomaly_window_count": 1,
                "held_out_normal_window_count": 1,
                "held_out_total_window_count": 2,
            },
        ]

        summary = _summarize_by_condition(rows, bootstrap_samples=100)
        delta_rows = _paired_delta_rows(rows, bootstrap_samples=100)

        self.assertEqual({row["condition"] for row in summary}, {"fixed", "evidence"})
        fixed_row = next(row for row in summary if row["condition"] == "fixed")
        evidence_row = next(row for row in summary if row["condition"] == "evidence")
        self.assertEqual(fixed_row["heldout_support_gap_mean"], 1.0)
        self.assertEqual(evidence_row["heldout_support_gap_mean"], 0.0)
        self.assertTrue(any(row["condition"] == "fixed_minus_evidence" for row in delta_rows))
        delta_row = next(row for row in delta_rows if row.get("condition") == "fixed_minus_evidence" and row.get("summary_kind") == "paired_delta")
        self.assertEqual(delta_row["heldout_support_gap_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
