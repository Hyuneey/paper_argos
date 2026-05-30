from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_NAMES = [
    "88cf3a776ba00e7c.csv",
    "9ee5879409dccef9.csv",
]

CONDITIONS = [
    {
        "name": "fixed",
        "segment_selection_mode": "fixed",
        "segment_selector_config": "",
    },
    {
        "name": "random",
        "segment_selection_mode": "evidence",
        "segment_selector_config": "configs/segment_selector_random.yaml",
    },
    {
        "name": "anomaly_centered",
        "segment_selection_mode": "evidence",
        "segment_selector_config": "configs/segment_selector_anomaly_centered.yaml",
    },
    {
        "name": "event_bounded_reference",
        "segment_selection_mode": "evidence",
        "segment_selector_config": "configs/segment_selector_event_reference.yaml",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the C1 comparative experiment suite.")
    parser.add_argument(
        "--dataset_paths",
        nargs="+",
        default=None,
        help="Explicit dataset CSVs. Defaults to the eligible KPI series.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="results/datasets/kpi_preliminary",
        help="Directory containing eligible KPI CSVs.",
    )
    parser.add_argument(
        "--output_root",
        default="experiments/c1_comparison_suite",
        help="Root directory for suite outputs.",
    )
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=[250, 1000, 2500])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--sample_per_prompt", type=int, default=1)
    parser.add_argument("--llm_engine", default="gpt-4-mini")
    parser.add_argument(
        "--llm_provider",
        choices=["azure", "openai", "chatgpt-oauth", "self-hosted", "auto"],
        default="chatgpt-oauth",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--retry_wait_sec", type=int, default=15)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_paths = _resolve_dataset_paths(repo_root, args.dataset_paths, args.dataset_dir)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for key in (
        "OPENAI_API_KEY",
        "ARGOS_OPENAI_OAUTH_TOKEN",
        "OPENAI_OAUTH_TOKEN",
        "ARGOS_CODEX_AUTH_PATH",
        "OPENAI_BASE_URL",
        "ARGOS_OPENAI_BASE_URL",
        "OPENAI_AZURE_API_KEY",
        "OPENAI_AZURE_ENDPOINT",
        "OPENAI_AZURE_API_VERSION",
    ):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    for dataset_path in dataset_paths:
        stem = Path(dataset_path).stem
        for condition in CONDITIONS:
            condition_root = output_root / condition["name"] / stem
            condition_root.mkdir(parents=True, exist_ok=True)
            command = [
                args.python,
                str(repo_root / "scripts" / "run_chunk_sensitivity.py"),
                "--dataset_path",
                str(dataset_path),
                "--output_root",
                str(condition_root),
                "--chunk_sizes",
                *[str(value) for value in args.chunk_sizes],
                "--repeats",
                str(args.repeats),
                "--top_k",
                str(args.top_k),
                "--sample_per_prompt",
                str(args.sample_per_prompt),
                "--llm_engine",
                args.llm_engine,
                "--llm_provider",
                args.llm_provider,
                "--temperature",
                str(args.temperature),
                "--timeout",
                str(args.timeout),
                "--seed",
                str(args.seed),
                "--max_attempts",
                str(args.max_attempts),
                "--retry_wait_sec",
                str(args.retry_wait_sec),
                "--segment_selection_mode",
                condition["segment_selection_mode"],
            ]
            if args.max_iter is not None:
                command.extend(["--max_iter", str(args.max_iter)])
            if condition["segment_selector_config"]:
                command.extend(["--segment_selector_config", str(repo_root / condition["segment_selector_config"])])
            print("[C1]", " ".join(command))
            if args.dry_run:
                continue
            subprocess.run(command, cwd=repo_root, env=env, check=True)


def _resolve_dataset_paths(repo_root: Path, dataset_paths: list[str] | None, dataset_dir: str) -> list[Path]:
    if dataset_paths:
        paths = [Path(path) for path in dataset_paths]
    else:
        paths = [Path(dataset_dir) / name for name in DEFAULT_DATASET_NAMES]
    resolved = []
    for path in paths:
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        resolved.append(path)
    return resolved


if __name__ == "__main__":
    main()
