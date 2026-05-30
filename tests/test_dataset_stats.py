import tempfile
from pathlib import Path
import unittest

import pandas as pd

from datasets.dataset import ArgosDataset


class DatasetSplitStatsTests(unittest.TestCase):
    def test_split_stats_counts_points_events_and_flags_small_test_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "toy.csv"
            df = pd.DataFrame(
                {
                    "value": list(range(10)),
                    "label": [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                    "index": list(range(10)),
                }
            )
            df.to_csv(csv_path, index=False)

            dataset = ArgosDataset(str(csv_path), dataset_mode="one-by-one", chunk_size=4)
            stats = dataset.get_split_anomaly_stats()

            self.assertEqual(stats["train"]["anomaly_point_count"], 2)
            self.assertEqual(stats["train"]["anomaly_event_count"], 1)
            self.assertAlmostEqual(stats["train"]["anomaly_point_ratio"], 2 / 5)
            self.assertEqual(stats["val"]["anomaly_point_count"], 1)
            self.assertEqual(stats["val"]["anomaly_event_count"], 1)
            self.assertEqual(stats["test"]["anomaly_point_count"], 2)
            self.assertEqual(stats["test"]["anomaly_event_count"], 1)
            self.assertTrue(stats["flags"]["test_event_count_lt_5"])
