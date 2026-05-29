import argparse
import csv
import json
import re
from pathlib import Path


SUMMARY_COLUMNS = [
    "dataset",
    "chunk_size",
    "repeat_id",
    "train_f1",
    "val_f1",
    "test_f1",
    "precision",
    "recall",
    "point_f1",
    "point_f1pa",
    "event_f1pa",
    "rule_num_lines",
    "rule_num_conditions",
    "rule_num_thresholds",
    "prompt_rows",
    "avg_chunk_anomaly_density",
    "max_chunk_anomaly_density",
    "train_val_gap",
    "val_test_gap",
    "runtime_sec",
    "token_count_detection",
    "token_count_repair",
    "token_count_review",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate ARGOS chunk sensitivity experiment outputs."
    )
    parser.add_argument(
        "--experiments_root",
        default="experiments/chunk_sensitivity",
    )
    parser.add_argument(
        "--output_csv",
        default="results/chunk_sensitivity_summary.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        experiments_root = repo_root / experiments_root
    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(experiments_root.glob("chunk_*/run_*")):
        if not run_dir.is_dir():
            continue
        rows.append(aggregate_run(run_dir))
    rows.sort(key=lambda row: (row.get("dataset") or "", row.get("chunk_size") or 0, row.get("repeat_id") or 0))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_run(run_dir: Path) -> dict:
    metadata = _read_json(run_dir / "metadata.json")
    stats = _read_json(run_dir / "stats.json")
    chunk_size = _parse_chunk_size(run_dir) or metadata.get("chunk_size")
    repeat_id = _parse_repeat_id(run_dir) or metadata.get("repeat_id")

    rule_path = _resolve_best_rule_path(run_dir, stats)
    train_eval = _read_eval_for_rule(rule_path, "train") or _best_eval(run_dir, "train")
    val_eval = _read_eval_for_rule(rule_path, "val") or _best_eval(run_dir, "val")
    test_eval = _read_eval_for_rule(rule_path, "test") or _best_eval(run_dir, "test")
    rule_text = rule_path.read_text(encoding="utf-8") if rule_path and rule_path.exists() else ""
    densities = _trace_densities(run_dir)

    train_f1 = _metric_value(train_eval, "f1")
    val_f1 = _metric_value(val_eval, "f1")
    test_f1 = _metric_value(test_eval, "f1")
    event_test = _metric_block(test_eval, "event-based f1 under pa with mode squeeze")
    point_test = _metric_block(test_eval, "point-wise f1")
    point_pa_test = _metric_block(test_eval, "best f1 under pa") or _metric_block(
        test_eval, "point-wise f1 pa"
    )

    return {
        "dataset": metadata.get("dataset"),
        "chunk_size": chunk_size,
        "repeat_id": repeat_id,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "precision": _metric_value(event_test or test_eval, "precision"),
        "recall": _metric_value(event_test or test_eval, "recall"),
        "point_f1": _metric_value(point_test, "f1"),
        "point_f1pa": _metric_value(point_pa_test, "f1"),
        "event_f1pa": _metric_value(event_test, "f1") or test_f1,
        "rule_num_lines": _rule_num_lines(rule_text),
        "rule_num_conditions": _count_regex(rule_text, r"\b(if|elif)\b"),
        "rule_num_thresholds": _count_regex(rule_text, r"(?<![A-Za-z_])\d+(?:\.\d+)?"),
        "prompt_rows": _prompt_rows(run_dir),
        "avg_chunk_anomaly_density": _avg(densities),
        "max_chunk_anomaly_density": max(densities) if densities else None,
        "train_val_gap": _gap(train_f1, val_f1),
        "val_test_gap": _gap(val_f1, test_f1),
        "runtime_sec": stats.get("time_elapsed"),
        "token_count_detection": _agent_token_total(stats, "DetectionAgentV3"),
        "token_count_repair": _agent_token_total(stats, "RepairAgent"),
        "token_count_review": _agent_token_total(stats, "ReviewAgent"),
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_best_rule_path(run_dir: Path, stats: dict) -> Path | None:
    best_paths = stats.get("best_rule_paths") or []
    if isinstance(best_paths, str):
        best_paths = [best_paths]
    if best_paths:
        candidate = Path(best_paths[-1])
        if candidate.exists():
            return candidate
        candidate = run_dir / candidate.name
        if candidate.exists():
            return candidate

    best_file = run_dir / "best_rule_path.txt"
    if best_file.exists():
        lines = [line.strip() for line in best_file.read_text().splitlines() if line.strip()]
        if lines:
            candidate = Path(lines[-1])
            if candidate.exists():
                return candidate
            candidate = run_dir / candidate.name
            if candidate.exists():
                return candidate

    rules = sorted(run_dir.glob("rule*.py"))
    return rules[-1] if rules else None


def _read_eval_for_rule(rule_path: Path | None, split: str) -> dict:
    if not rule_path:
        return {}
    return _read_json(rule_path.with_name(rule_path.stem + f"_eval_res_{split}.json"))


def _best_eval(run_dir: Path, split: str) -> dict:
    evals = [_read_json(path) for path in run_dir.glob(f"*eval_res_{split}.json")]
    evals = [item for item in evals if item]
    if not evals:
        return {}
    return max(evals, key=lambda item: _metric_value(item, "f1") or 0.0)


def _metric_block(metrics: dict, name: str) -> dict:
    value = metrics.get(name)
    return value if isinstance(value, dict) else {}


def _metric_value(metrics: dict, key: str):
    if not metrics:
        return None
    if key in metrics and not isinstance(metrics[key], dict):
        return metrics[key]
    for value in metrics.values():
        if isinstance(value, dict) and key in value:
            return value[key]
    return None


def _agent_token_total(stats: dict, agent_name: str):
    token_count = stats.get("token_count", {}).get(agent_name)
    if isinstance(token_count, list) and len(token_count) >= 3:
        return token_count[1] + token_count[2]
    if isinstance(token_count, tuple) and len(token_count) >= 3:
        return token_count[1] + token_count[2]
    return None


def _trace_densities(run_dir: Path) -> list[float]:
    densities = []
    for trace_path in run_dir.glob("selection_trace_iter_*.json"):
        trace = _read_json(trace_path)
        value = trace.get("selection_score", {}).get("anomaly_density")
        if value is not None:
            densities.append(value)
    return densities


def _prompt_rows(run_dir: Path):
    metadata = _read_json(run_dir / "metadata.json")
    top_k = metadata.get("top_k") or 1
    trace_paths = list(run_dir.glob("selection_trace_iter_*.json"))
    if trace_paths:
        return top_k * sum(
            (_read_json(path).get("selected_segment", {}).get("length") or 0)
            for path in trace_paths
        )
    stats = _read_json(run_dir / "stats.json")
    iterations = stats.get("cur_iter") or 0
    return (metadata.get("chunk_size") or 0) * iterations * top_k


def _rule_num_lines(rule_text: str) -> int:
    return len([line for line in rule_text.splitlines() if line.strip()])


def _count_regex(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def _avg(values):
    return sum(values) / len(values) if values else None


def _gap(left, right):
    if left is None or right is None:
        return None
    return left - right


def _parse_chunk_size(run_dir: Path):
    match = re.search(r"chunk_(\d+)", str(run_dir))
    return int(match.group(1)) if match else None


def _parse_repeat_id(run_dir: Path):
    match = re.search(r"run_(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    main()
