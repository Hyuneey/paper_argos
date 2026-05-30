import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_morphology_diagnostic import _build_morphology_table, _merge_diagnostics, _summarize_rows, main as build_main


class MorphologyDiagnosticTests(unittest.TestCase):
    def test_build_morphology_table_computes_event_shape_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "datasets"
            dataset_dir.mkdir()
            pd.DataFrame(
                {
                    "value": [0.0, 1.0, 2.0, 0.0, 10.0, 10.5, 11.0, 0.0],
                    "label": [0, 1, 1, 0, 1, 1, 1, 0],
                    "index": list(range(8)),
                }
            ).to_csv(dataset_dir / "alpha.csv", index=False)

            rows = _build_morphology_table(dataset_dir)

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["dataset"], "alpha")
            self.assertEqual(row["anomaly_event_count"], 2)
            self.assertEqual(row["morphology_label"], "spike")
            self.assertAlmostEqual(row["mean_event_len"], 2.5)
            self.assertGreater(row["anomaly_point_ratio"], 0.0)

    def test_merge_diagnostics_combines_performance_and_support_gap(self):
        morphology_rows = [
            {
                "dataset": "alpha",
                "total_points": 8,
                "anomaly_point_count": 5,
                "anomaly_event_count": 2,
                "anomaly_point_ratio": 0.625,
                "anomaly_event_frequency_per_1k": 250.0,
                "mean_event_len": 2.5,
                "median_event_len": 2.5,
                "max_event_len": 3.0,
                "mean_event_amplitude": 3.5,
                "max_event_amplitude": 6.0,
                "mean_event_abs_slope": 1.0,
                "max_event_abs_slope": 1.5,
                "morphology_label": "spike",
            }
        ]
        perf_summary = {
            ("alpha", "fixed"): {"rows": 2, "mean_test_f1": 0.8, "mean_point_f1_fixed": 0.75, "mean_point_f1": 0.7, "mean_event_f1pa": 0.6, "mean_token_count_detection": 100.0, "mean_prompt_rows": 20.0},
            ("alpha", "evidence"): {"rows": 2, "mean_test_f1": 0.6, "mean_point_f1_fixed": 0.55, "mean_point_f1": 0.5, "mean_event_f1pa": 0.4, "mean_token_count_detection": 80.0, "mean_prompt_rows": 15.0},
        }
        cons_summary = {
            ("alpha", "fixed"): {"mean_heldout_support_gap": 0.3, "mean_local_support_gap": 0.25},
            ("alpha", "evidence"): {"mean_heldout_support_gap": 0.1, "mean_local_support_gap": 0.05},
        }

        merged = _merge_diagnostics(morphology_rows, perf_summary, cons_summary)
        self.assertEqual(len(merged), 1)
        row = merged[0]
        self.assertEqual(row["fixed_rows"], 2)
        self.assertEqual(row["evidence_rows"], 2)
        self.assertAlmostEqual(row["delta_mean_test_f1"], 0.2)
        self.assertAlmostEqual(row["delta_mean_heldout_support_gap"], 0.2)
        self.assertEqual(row["morphology_label"], "spike")

    def test_cli_writes_table_and_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "datasets"
            dataset_dir.mkdir()
            pd.DataFrame(
                {
                    "value": [0.0, 1.0, 2.0, 0.0],
                    "label": [0, 1, 1, 0],
                    "index": list(range(4)),
                }
            ).to_csv(dataset_dir / "alpha.csv", index=False)

            perf_csv = root / "perf.csv"
            perf_csv.write_text(
                "dataset,condition,test_f1,point_f1_fixed,point_f1,event_f1pa,runtime_sec,token_count_detection,prompt_rows\n"
                "alpha,fixed,0.8,0.75,0.7,0.6,1.0,100,20\n"
                "alpha,evidence,0.6,0.55,0.5,0.4,1.0,80,15\n",
                encoding="utf-8",
            )
            cons_csv = root / "cons.csv"
            cons_csv.write_text(
                "dataset,condition,heldout_support_gap,local_support_gap\n"
                "alpha,fixed,0.3,0.25\n"
                "alpha,evidence,0.1,0.05\n",
                encoding="utf-8",
            )
            output_csv = root / "diag.csv"
            note_path = root / "note.md"

            old_argv = list(__import__("sys").argv)
            try:
                __import__("sys").argv = [
                    "build_morphology_diagnostic.py",
                    "--dataset_dir",
                    str(dataset_dir),
                    "--performance_rows_csv",
                    str(perf_csv),
                    "--comparison_rows_csv",
                    str(cons_csv),
                    "--output_csv",
                    str(output_csv),
                    "--note_path",
                    str(note_path),
                ]
                build_main()
            finally:
                __import__("sys").argv = old_argv

            self.assertTrue(output_csv.exists())
            self.assertTrue(note_path.exists())
            self.assertIn("morphology_label", output_csv.read_text(encoding="utf-8"))
            self.assertIn("Highest finite support-gap delta", note_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
