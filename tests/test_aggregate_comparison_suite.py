import csv
import tempfile
from pathlib import Path
import unittest

from scripts.aggregate_comparison_suite import _best_fixed_selected_rows, _write_csv, _summarize_conditions, _fixed_vs_evidence_delta_rows


class AggregateComparisonSuiteTests(unittest.TestCase):
    def test_best_fixed_selected_rows_picks_highest_val_chunk(self):
        rows = [
            {
                "condition": "fixed",
                "series": "alpha",
                "dataset": "alpha",
                "chunk_size": 100,
                "val_f1": 0.20,
                "test_f1": 0.10,
                "point_f1_fixed": 0.11,
                "point_f1": 0.12,
                "event_f1pa": 0.13,
                "runtime_sec": 10.0,
            },
            {
                "condition": "fixed",
                "series": "alpha",
                "dataset": "alpha",
                "chunk_size": 100,
                "val_f1": 0.40,
                "test_f1": 0.30,
                "point_f1_fixed": 0.31,
                "point_f1": 0.32,
                "event_f1pa": 0.33,
                "runtime_sec": 12.0,
            },
            {
                "condition": "fixed",
                "series": "alpha",
                "dataset": "alpha",
                "chunk_size": 250,
                "val_f1": 0.90,
                "test_f1": 0.70,
                "point_f1_fixed": 0.71,
                "point_f1": 0.72,
                "event_f1pa": 0.73,
                "runtime_sec": 14.0,
            },
        ]

        selected_rows = _best_fixed_selected_rows(rows)

        self.assertEqual(len(selected_rows), 1)
        selected = selected_rows[0]
        self.assertEqual(selected["condition"], "best_fixed_val")
        self.assertEqual(selected["series"], "alpha")
        self.assertEqual(selected["chunk_size"], 250)
        self.assertEqual(selected["selection_key"], "mean_val_f1")
        self.assertEqual(selected["selection_value"], 0.9)
        self.assertEqual(selected["rows"], 1)

    def test_summarize_conditions_includes_main_report_metrics(self):
        rows = [
            {
                "condition": "fixed",
                "dataset": "alpha",
                "val_f1": 0.2,
                "test_f1": 0.3,
                "point_f1_fixed": 0.4,
                "point_f1": 0.5,
                "event_f1pa": 0.6,
                "runtime_sec": 10.0,
                "token_count_detection": 100.0,
                "prompt_rows": 50.0,
            },
            {
                "condition": "fixed",
                "dataset": "alpha",
                "val_f1": 0.4,
                "test_f1": 0.6,
                "point_f1_fixed": 0.8,
                "point_f1": 0.9,
                "event_f1pa": 1.0,
                "runtime_sec": 14.0,
                "token_count_detection": 200.0,
                "prompt_rows": 70.0,
            },
        ]

        summary = _summarize_conditions(rows)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["condition"], "fixed")
        self.assertEqual(row["dataset"], "alpha")
        self.assertAlmostEqual(row["mean_point_f1_fixed"], 0.6)
        self.assertAlmostEqual(row["mean_point_f1"], 0.7)
        self.assertAlmostEqual(row["mean_event_f1pa"], 0.8)
        self.assertAlmostEqual(row["std_point_f1"], 0.28284271247461906)
        self.assertAlmostEqual(row["ci95_event_f1pa"], 0.392)

    def test_fixed_vs_evidence_delta_rows_compute_fixed_minus_evidence(self):
        rows = [
            {
                "condition": "fixed",
                "dataset": "alpha",
                "series": "alpha",
                "chunk_size": 100,
                "point_f1_fixed": 0.6,
                "point_f1": 0.7,
                "event_f1pa": 0.8,
                "runtime_sec": 10.0,
                "token_count_detection": 100.0,
                "prompt_rows": 50.0,
            },
            {
                "condition": "fixed",
                "dataset": "alpha",
                "series": "alpha",
                "chunk_size": 100,
                "point_f1_fixed": 0.8,
                "point_f1": 0.9,
                "event_f1pa": 1.0,
                "runtime_sec": 14.0,
                "token_count_detection": 200.0,
                "prompt_rows": 70.0,
            },
            {
                "condition": "evidence",
                "dataset": "alpha",
                "series": "alpha",
                "chunk_size": 100,
                "point_f1_fixed": 0.2,
                "point_f1": 0.3,
                "event_f1pa": 0.4,
                "runtime_sec": 6.0,
                "token_count_detection": 20.0,
                "prompt_rows": 10.0,
            },
            {
                "condition": "evidence",
                "dataset": "alpha",
                "series": "alpha",
                "chunk_size": 100,
                "point_f1_fixed": 0.4,
                "point_f1": 0.5,
                "event_f1pa": 0.6,
                "runtime_sec": 8.0,
                "token_count_detection": 40.0,
                "prompt_rows": 30.0,
            },
        ]

        delta_rows = _fixed_vs_evidence_delta_rows(rows)

        self.assertEqual(len(delta_rows), 1)
        row = delta_rows[0]
        self.assertEqual(row["dataset"], "alpha")
        self.assertEqual(row["series"], "alpha")
        self.assertEqual(row["chunk_size"], 100)
        self.assertEqual(row["fixed_rows"], 2)
        self.assertEqual(row["evidence_rows"], 2)
        self.assertAlmostEqual(row["fixed_point_f1_fixed"], 0.7)
        self.assertAlmostEqual(row["evidence_point_f1_fixed"], 0.3)
        self.assertAlmostEqual(row["delta_point_f1_fixed"], 0.4)
        self.assertAlmostEqual(row["delta_point_f1"], 0.4)
        self.assertAlmostEqual(row["delta_event_f1pa"], 0.4)
        self.assertAlmostEqual(row["delta_runtime_sec"], 5.0)
        self.assertAlmostEqual(row["delta_token_count_detection"], 120.0)
        self.assertAlmostEqual(row["delta_prompt_rows"], 40.0)

    def test_write_csv_dedupes_header_fields(self):
        rows = [
            {
                "condition": "fixed",
                "series": "alpha",
                "suite_root": "/tmp/suite",
                "dataset": "alpha",
                "chunk_size": 250,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "rows.csv"
            _write_csv(
                out_path,
                rows,
                ["condition", "series", "suite_root", "condition", "series", "dataset", "chunk_size"],
            )

            header = out_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                header,
                "condition,series,suite_root,dataset,chunk_size",
            )

            with out_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                parsed = list(reader)
            self.assertEqual(parsed[0]["dataset"], "alpha")
            self.assertEqual(parsed[0]["chunk_size"], "250")
