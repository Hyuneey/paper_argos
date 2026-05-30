import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CHUNK_SIZES = [100, 250, 500, 1000, 2500, 5000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ARGOS chunk-size sensitivity experiments."
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--output_root",
        default="experiments/chunk_sensitivity",
        help="Root directory for chunk sensitivity runs.",
    )
    parser.add_argument(
        "--chunk_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_CHUNK_SIZES,
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mode", default="train-LLM-only")
    parser.add_argument("--dataset_mode", default="one-by-one")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_per_prompt", type=int, default=1)
    parser.add_argument("--llm_engine", default="gpt-4o-mini")
    parser.add_argument(
        "--llm_provider",
        choices=["azure", "openai", "chatgpt-oauth", "self-hosted", "auto"],
        default="openai",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--retry_wait_sec", type=int, default=15)
    parser.add_argument(
        "--segment_selection_mode",
        choices=["fixed", "evidence"],
        default="fixed",
    )
    parser.add_argument(
        "--segment_selector_config",
        default="configs/segment_selector_default.yaml",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run driver.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    selector_config = args.segment_selector_config
    if selector_config and not Path(selector_config).is_absolute():
        selector_config = str(repo_root / selector_config)

    for chunk_size in args.chunk_sizes:
        for repeat_id in range(1, args.repeats + 1):
            run_dir = output_root / f"chunk_{chunk_size}" / f"run_{repeat_id:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "stats.json").exists():
                continue

            command = [
                args.python,
                str(repo_root / "driver.py"),
                "--dataset_path",
                args.dataset_path,
                "--mode",
                args.mode,
                "--result_path",
                str(run_dir),
                "--result_path_is_final",
                "--chunk_size",
                str(chunk_size),
                "--top_k",
                str(args.top_k),
                "--dataset_mode",
                args.dataset_mode,
                "--repeat",
                "1",
                "--sample_per_prompt",
                str(args.sample_per_prompt),
                "--llm_engine",
                args.llm_engine,
                "--llm_provider",
                args.llm_provider,
                "--temperature",
                str(args.temperature),
                "--seed",
                str(args.seed),
                "--timeout",
                str(args.timeout),
                "--segment_selection_mode",
                args.segment_selection_mode,
            ]
            if args.max_iter is not None:
                command.extend(["--max_iter", str(args.max_iter)])
            if args.segment_selection_mode == "evidence" and selector_config:
                command.extend(["--segment_selector_config", selector_config])

            log_path = run_dir / "driver_stdout.log"
            last_error = None
            for attempt_id in range(1, args.max_attempts + 1):
                _cleanup_incomplete_run(run_dir)
                attempt_log_path = run_dir / f"driver_stdout_attempt_{attempt_id:02d}.log"
                with open(attempt_log_path, "w", encoding="utf-8") as log_file:
                    try:
                        child_env = os.environ.copy()
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
                                child_env[key] = value
                        subprocess.run(
                            command,
                            cwd=repo_root,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            check=True,
                            env=child_env,
                        )
                        shutil.copyfile(attempt_log_path, log_path)
                        last_error = None
                        break
                    except subprocess.CalledProcessError as exc:
                        last_error = exc
                if attempt_id < args.max_attempts:
                    time.sleep(args.retry_wait_sec)
            if last_error is not None:
                raise last_error


def _cleanup_incomplete_run(run_dir: Path) -> None:
    generated_patterns = [
        "rule*.py",
        "*_eval_res_train.json",
        "*_eval_res_val.json",
        "*_eval_res_test.json",
        "stats.json",
        "metadata.json",
        "output.log",
        "driver_stdout.log",
        "driver_stdout_attempt_*.log",
        "selection_trace_iter_*.json",
        "best_rule_path.txt",
    ]
    for pattern in generated_patterns:
        for path in run_dir.glob(pattern):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
