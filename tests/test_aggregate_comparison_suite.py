import csv
import tempfile
from pathlib import Path
import unittest

from scripts.aggregate_comparison_suite import _best_fixed_selected_rows, _write_csv


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
