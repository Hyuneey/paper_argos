import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from datasets.dataset import ArgosDataset
from segment_selection.candidate_generator import generate_candidates
from segment_selection.selector import SegmentSelector
from segment_selection.trace_logger import write_selection_trace
from segment_selection.utility_scorer import load_selector_config


class SegmentSelectionTests(unittest.TestCase):
    def test_fixed_mode_preserves_chunking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "series.csv"
            _sample_df(20).to_csv(path, index=False)

            dataset = ArgosDataset(str(path), chunk_size=5, segment_selection_mode="fixed")

            self.assertEqual(len(dataset.train_dict), 3)
            pd.testing.assert_frame_equal(
                dataset.get_train_df_by_iter(0).reset_index(drop=True),
                dataset.train_dict[0].reset_index(drop=True),
            )

    def test_evidence_mode_returns_segment_and_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "series.csv"
            _sample_df(30, anomaly_ranges=[(10, 13)]).to_csv(path, index=False)

            dataset = ArgosDataset(
                str(path),
                chunk_size=10,
                segment_selection_mode="evidence",
                selection_trace_dir=tmpdir,
            )
            selected = dataset.get_train_df_by_iter(0)

            self.assertGreater(len(selected), 0)
            self.assertTrue(dataset.get_selection_trace_paths())
            self.assertTrue(list(Path(tmpdir).glob("selection_trace_iter_*.json")))

    def test_candidate_generator_handles_edges_and_no_anomaly(self):
        df = _sample_df(12, anomaly_ranges=[(0, 2), (10, 12)])
        candidates = generate_candidates(df, chunk_size=6, iter_num=1)

        kinds = {candidate.kind for candidate in candidates}
        self.assertIn("fixed_chunk", kinds)
        self.assertIn("short_anomaly_centered", kinds)
        self.assertTrue(all(candidate.length > 0 for candidate in candidates))
        self.assertTrue(all(0 <= candidate.start_pos <= candidate.end_pos <= len(df) for candidate in candidates))

        no_anomaly_candidates = generate_candidates(_sample_df(10), chunk_size=4, iter_num=2)
        self.assertEqual(
            {candidate.kind for candidate in no_anomaly_candidates},
            {"fixed_chunk", "random_segment"},
        )

    def test_utility_scorer_is_deterministic_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "selector.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "weights:",
                        "  anomaly_density: 0.30",
                        "  change_magnitude: 0.20",
                        "  anomaly_coverage: 0.20",
                        "  normal_contrast: 0.15",
                        "  length_penalty: 0.10",
                        "  token_cost: 0.05",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_selector_config(str(config_path))
            self.assertEqual(config["weights"]["anomaly_density"], 0.30)

            df = _sample_df(20, anomaly_ranges=[(8, 11)])
            candidates = generate_candidates(df, chunk_size=10, iter_num=3)
            selector = SegmentSelector(str(config_path))
            first = selector.select(candidates, df, 10)
            second = selector.select(candidates, df, 10)

            self.assertEqual(first.selected.kind, second.selected.kind)
            self.assertEqual(first.selected_score.total, second.selected_score.total)

    def test_trace_logger_writes_expected_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = _sample_df(20, anomaly_ranges=[(7, 9)])
            candidates = generate_candidates(df, chunk_size=10, iter_num=4)
            result = SegmentSelector().select(candidates, df, 10)

            trace_path = write_selection_trace(
                result,
                tmpdir,
                iter_num=4,
                call_id=0,
                random_seed=8,
                config_path=None,
            )

            payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            self.assertIn("selected_segment", payload)
            self.assertIn("selection_score", payload)
            self.assertIn("candidate_scores", payload)
            self.assertTrue(payload["oracle_like_analysis_setting"])


def _sample_df(length, anomaly_ranges=None):
    labels = [0] * length
    for start, end in anomaly_ranges or []:
        for idx in range(start, min(end, length)):
            labels[idx] = 1
    return pd.DataFrame(
        {
            "value": [float(idx % 5) for idx in range(length)],
            "label": labels,
            "index": list(range(length)),
        }
    )


if __name__ == "__main__":
    unittest.main()
