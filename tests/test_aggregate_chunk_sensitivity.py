import json
import tempfile
from pathlib import Path
import unittest

from scripts.aggregate_chunk_sensitivity import (
    aggregate_run,
    summarize_rows_by_group,
    _flatten_split_stats,
)


class AggregateChunkSensitivityTests(unittest.TestCase):
    def test_flatten_split_stats_expands_expected_split_fields(self):
        split_stats = {
            "train": {
                "total_points": 10,
                "anomaly_point_count": 2,
                "anomaly_event_count": 1,
                "anomaly_point_ratio": 0.2,
                "anomaly_event_ratio": 0.1,
            },
            "val": {
                "total_points": 5,
                "anomaly_point_count": 1,
                "anomaly_event_count": 1,
                "anomaly_point_ratio": 0.2,
                "anomaly_event_ratio": 0.2,
            },
            "test": {
                "total_points": 8,
                "anomaly_point_count": 3,
                "anomaly_event_count": 2,
                "anomaly_point_ratio": 0.375,
                "anomaly_event_ratio": 0.25,
            },
            "flags": {
                "test_event_count_lt_5": True,
                "test_anomaly_point_count_lt_5": False,
            },
        }

        flat = _flatten_split_stats(split_stats)

        self.assertEqual(flat["split_train_total_points"], 10)
        self.assertEqual(flat["split_val_anomaly_event_count"], 1)
        self.assertEqual(flat["split_test_anomaly_point_ratio"], 0.375)
        self.assertTrue(flat["split_flags_test_event_count_lt_5"])
        self.assertFalse(flat["split_flags_test_anomaly_point_count_lt_5"])

    def test_aggregate_run_includes_flattened_split_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rule_path = run_dir / "rule_iter0_0.py"
            rule_path.write_text(
                "if x > 0:\n    return 1\n",
                encoding="utf-8",
            )
            for split, payload in {
                "train": {"f1": 0.1},
                "val": {"f1": 0.2},
                "test": {
                    "f1": 0.3,
                    "precision": 0.8,
                    "recall": 0.7,
                    "point-wise f1": {"f1": 0.4},
                    "point-wise fixed f1": {"f1": 0.5},
                    "best f1 under pa": {"f1": 0.6},
                    "event-based f1 under pa with mode squeeze": {
                        "f1": 0.7,
                        "precision": 0.8,
                        "recall": 0.9,
                    },
                    "affiliation f1": {"f1": 0.55},
                },
            }.items():
                (run_dir / f"rule_iter0_0_eval_res_{split}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            metadata = {
                "dataset": "toy",
                "chunk_size": 100,
                "repeat_id": 1,
                "segment_selection_mode": "fixed",
                "llm_provider": "openai",
                "llm_engine": "gpt-4o-mini",
                "temperature": 0.0,
                "top_k": 5,
                "max_iter": 1,
                "seed": 8,
                "split_stats": {
                    "train": {
                        "total_points": 10,
                        "anomaly_point_count": 2,
                        "anomaly_event_count": 1,
                        "anomaly_point_ratio": 0.2,
                        "anomaly_event_ratio": 0.1,
                    },
                    "val": {
                        "total_points": 5,
                        "anomaly_point_count": 1,
                        "anomaly_event_count": 1,
                        "anomaly_point_ratio": 0.2,
                        "anomaly_event_ratio": 0.2,
                    },
                    "test": {
                        "total_points": 8,
                        "anomaly_point_count": 3,
                        "anomaly_event_count": 2,
                        "anomaly_point_ratio": 0.375,
                        "anomaly_event_ratio": 0.25,
                    },
                    "flags": {
                        "test_event_count_lt_5": True,
                        "test_anomaly_point_count_lt_5": False,
                    },
                },
            }
            stats = {
                "time_elapsed": 12.5,
                "best_rule_paths": [str(rule_path)],
                "token_count": {
                    "DetectionAgentV3": [0, 11, 13],
                    "RepairAgent": [0, 5, 7],
                    "ReviewAgent": [0, 2, 3],
                },
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (run_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

            row = aggregate_run(run_dir)

            self.assertIsNotNone(row)
            self.assertEqual(row["dataset"], "toy")
            self.assertEqual(row["segment_selection_mode"], "fixed")
            self.assertEqual(row["llm_engine"], "gpt-4o-mini")
            self.assertEqual(row["max_iter"], 1)
            self.assertEqual(row["seed"], 8)
            self.assertEqual(row["point_f1"], 0.5)
            self.assertEqual(row["point_f1_fixed"], 0.5)
            self.assertEqual(row["point_f1_oracle"], 0.4)
            self.assertEqual(row["split_train_total_points"], 10)
            self.assertEqual(row["split_test_anomaly_event_count"], 2)
            self.assertTrue(row["split_flags_test_event_count_lt_5"])
            self.assertEqual(row["runtime_sec"], 12.5)
            self.assertEqual(row["token_count_detection"], 24)
            self.assertEqual(row["token_count_review"], 5)

    def test_summarize_rows_by_group_emits_mean_std_and_ci(self):
        rows = [
            {
                "dataset": "toy",
                "segment_selection_mode": "fixed",
                "llm_provider": "openai",
                "llm_engine": "gpt-4o-mini",
                "temperature": 0.0,
                "top_k": 5,
                "max_iter": 1,
                "seed": 8,
                "chunk_size": 100,
                "repeat_id": 1,
                "point_f1_fixed": 0.25,
                "runtime_sec": 10.0,
            },
            {
                "dataset": "toy",
                "segment_selection_mode": "fixed",
                "llm_provider": "openai",
                "llm_engine": "gpt-4o-mini",
                "temperature": 0.0,
                "top_k": 5,
                "max_iter": 1,
                "seed": 8,
                "chunk_size": 100,
                "repeat_id": 2,
                "point_f1_fixed": 0.25,
                "runtime_sec": 10.0,
            },
        ]

        grouped = summarize_rows_by_group(rows)

        self.assertEqual(len(grouped), 1)
        grouped_row = grouped[0]
        self.assertEqual(grouped_row["n_repeats"], 2)
        self.assertEqual(grouped_row["point_f1_fixed_mean"], 0.25)
        self.assertEqual(grouped_row["point_f1_fixed_std"], 0.0)
        self.assertEqual(grouped_row["point_f1_fixed_ci_low"], 0.25)
        self.assertEqual(grouped_row["point_f1_fixed_ci_high"], 0.25)


if __name__ == "__main__":
    unittest.main()
