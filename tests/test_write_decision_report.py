import tempfile
import unittest
from pathlib import Path

from scripts.write_decision_report import main as write_decision_main


class DecisionReportTests(unittest.TestCase):
    def test_cli_writes_report_with_leg_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            perf = root / "perf.csv"
            perf.write_text(
                "condition,dataset,rows,mean_point_f1_fixed,mean_token_count_detection,mean_prompt_rows\n"
                "fixed,series1,3,0.20,100,20\n"
                "evidence,series1,3,0.30,80,15\n",
                encoding="utf-8",
            )
            cost = root / "cost.csv"
            cost.write_text(
                "dataset,condition,mean_token_count_detection,mean_prompt_rows\n"
                "series1,fixed,100,20\n"
                "series1,evidence,80,15\n",
                encoding="utf-8",
            )
            cons = root / "cons.csv"
            cons.write_text(
                "dataset,condition,heldout_support_gap_mean\n"
                "series1,fixed,0.10\n"
                "series1,evidence,0.20\n",
                encoding="utf-8",
            )
            morph = root / "morph.csv"
            morph.write_text(
                "dataset,morphology_label\n"
                "series1,spike\n",
                encoding="utf-8",
            )
            out = root / "DECISION.md"

            import sys

            old_argv = list(sys.argv)
            try:
                sys.argv = [
                    "write_decision_report.py",
                    "--performance_csv",
                    str(perf),
                    "--cost_csv",
                    str(cost),
                    "--consistency_csv",
                    str(cons),
                    "--morphology_csv",
                    str(morph),
                    "--output_path",
                    str(out),
                    "--commit_hash",
                    "abc123",
                ]
                write_decision_main()
            finally:
                sys.argv = old_argv

            text = out.read_text(encoding="utf-8")
            self.assertIn("Conclusion:", text)
            self.assertIn("Performance", text)
            self.assertIn("Cost", text)
            self.assertIn("Traceability", text)
            self.assertIn("abc123", text)
            self.assertIn("GO", text)

    def test_conclusion_becomes_go_reframe_for_mixed_legs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            perf = root / "perf.csv"
            perf.write_text(
                "condition,dataset,rows,mean_point_f1_fixed,mean_token_count_detection,mean_prompt_rows\n"
                "fixed,series1,3,0.30,100,20\n"
                "evidence,series1,3,0.20,80,15\n",
                encoding="utf-8",
            )
            cost = root / "cost.csv"
            cost.write_text(
                "dataset,condition,mean_token_count_detection,mean_prompt_rows\n"
                "series1,fixed,100,20\n"
                "series1,evidence,80,15\n",
                encoding="utf-8",
            )
            cons = root / "cons.csv"
            cons.write_text(
                "dataset,condition,heldout_support_gap_mean\n"
                "series1,fixed,0.10\n"
                "series1,evidence,0.05\n",
                encoding="utf-8",
            )
            morph = root / "morph.csv"
            morph.write_text("dataset,morphology_label\nseries1,mixed\n", encoding="utf-8")
            out = root / "DECISION.md"

            import sys

            old_argv = list(sys.argv)
            try:
                sys.argv = [
                    "write_decision_report.py",
                    "--performance_csv",
                    str(perf),
                    "--cost_csv",
                    str(cost),
                    "--consistency_csv",
                    str(cons),
                    "--morphology_csv",
                    str(morph),
                    "--output_path",
                    str(out),
                ]
                write_decision_main()
            finally:
                sys.argv = old_argv

            text = out.read_text(encoding="utf-8")
            self.assertIn("GO-reframe", text)
            self.assertIn("FAIL", text)


if __name__ == "__main__":
    unittest.main()
