import json
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.run_chunk_sensitivity import parse_args


def _load_driver_module():
    fake_tqdm = types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable)
    fake_openai = types.SimpleNamespace(AzureOpenAI=object, OpenAI=object)
    fake_tiktoken = types.SimpleNamespace(
        encoding_for_model=lambda model: object(),
        get_encoding=lambda name: object(),
    )
    fake_config = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            max_iter=1,
            timeout=1,
            timeout_first_review=1,
            timeout_llm=1,
            timeout_per_review=1,
            timeout_image=1,
            timeout_inference=1,
            self_hosted_llm_list=[],
        )
    )
    fake_runtime = types.ModuleType("runtime")
    fake_runtime_engine = types.ModuleType("runtime.engine")
    fake_runtime_engine.Engine = object
    fake_runtime.engine = fake_runtime_engine
    with patch.dict(
        "sys.modules",
        {
            "tqdm": fake_tqdm,
            "openai": fake_openai,
            "tiktoken": fake_tiktoken,
            "config": fake_config,
            "runtime": fake_runtime,
            "runtime.engine": fake_runtime_engine,
        },
    ):
        import importlib

        return importlib.import_module("driver")


DRIVER_MODULE = _load_driver_module()


class RuntimeConfigTests(unittest.TestCase):
    def test_run_chunk_sensitivity_defaults_to_pinned_public_model_and_five_repeats(self):
        argv = [
            "run_chunk_sensitivity.py",
            "--dataset_path",
            "/tmp/toy.csv",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.llm_provider, "openai")
        self.assertEqual(args.llm_engine, "gpt-4o-mini")
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.seed, 8)

    def test_driver_main_forwards_seed_into_engine_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                dataset_path="/tmp/toy.csv",
                result_path=tmpdir,
                result_path_is_final=True,
                rule_path="",
                mode="train-LLM-only",
                dataset_mode="one-by-one",
                train_test_split=0.7,
                model_res_path="",
                chunk_size=100,
                image_chunk_size=10000,
                image_subplots=(1, ",", 1),
                repeat=1,
                top_k=5,
                llm_engine="gpt-4o-mini",
                llm_provider="openai",
                temperature=0.0,
                timeout=150,
                sample_per_prompt=1,
                p_cores=5,
                rule_per_group=1,
                max_iter=None,
                seed=123,
                segment_selection_mode="fixed",
                segment_selector_config="configs/segment_selector_default.yaml",
            )

            mock_engine = types.SimpleNamespace(
                run=lambda: None,
                get_split_anomaly_stats=lambda: {"flags": {}},
            )
            with patch.object(DRIVER_MODULE, "Engine", return_value=mock_engine) as engine_ctor:
                with patch.object(DRIVER_MODULE.tqdm, "tqdm", side_effect=lambda iterable, **kwargs: iterable):
                    DRIVER_MODULE.main(args)

            engine_ctor.assert_called_once()
            self.assertEqual(engine_ctor.call_args.kwargs["seed"], 123)

            metadata = json.loads(Path(tmpdir, "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 123)
            self.assertEqual(metadata["temperature"], 0.0)
            self.assertEqual(metadata["llm_engine"], "gpt-4o-mini")
            self.assertEqual(metadata["resolved_llm_engine"], "gpt-4o-mini")

    def test_write_metadata_records_temperature_and_resolved_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                dataset_path="/tmp/toy.csv",
                chunk_size=100,
                mode="train-LLM-only",
                dataset_mode="one-by-one",
                top_k=1,
                repeat=5,
                llm_engine="gpt-4o-mini",
                llm_provider="openai",
                temperature=0.0,
                max_iter=1,
                seed=8,
                segment_selection_mode="fixed",
                segment_selector_config="configs/segment_selector_default.yaml",
            )

            DRIVER_MODULE.write_metadata(args, tmpdir, repeat_id=1, split_stats={"flags": {}})

            metadata = json.loads(Path(tmpdir, "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["temperature"], 0.0)
            self.assertEqual(metadata["max_iter"], 1)
            self.assertEqual(metadata["seed"], 8)
            self.assertEqual(metadata["resolved_llm_engine"], "gpt-4o-mini")
            self.assertEqual(metadata["llm_engine"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
