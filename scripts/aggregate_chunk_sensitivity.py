import argparse
import csv
import json
import random
import re
from pathlib import Path


SUMMARY_COLUMNS = [
    "dataset",
    "chunk_size",
    "segment_selection_mode",
    "llm_provider",
    "llm_engine",
    "temperature",
    "top_k",
    "max_iter",
    "seed",
    "repeat_id",
    "train_f1",
    "val_f1",
    "test_f1",
    "precision",
    "recall",
    "point_f1",
    "point_f1_fixed",
    "point_f1_oracle",
    "point_f1pa",
    "affiliation_f1",
    "event_f1pa",
    "split_train_total_points",
    "split_train_anomaly_point_count",
    "split_train_anomaly_event_count",
    "split_train_anomaly_point_ratio",
    "split_train_anomaly_event_ratio",
    "split_val_total_points",
    "split_val_anomaly_point_count",
    "split_val_anomaly_event_count",
    "split_val_anomaly_point_ratio",
    "split_val_anomaly_event_ratio",
    "split_test_total_points",
    "split_test_anomaly_point_count",
    "split_test_anomaly_event_count",
    "split_test_anomaly_point_ratio",
    "split_test_anomaly_event_ratio",
    "split_flags_test_event_count_lt_5",
    "split_flags_test_anomaly_point_count_lt_5",
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
        default="experiments",
    )
    parser.add_argument(
        "--output_csv",
        default="results/chunk_sensitivity_summary.csv",
    )
    parser.add_argument(
        "--grouped_output_csv",
        default="",
        help="Optional chunk-level summary CSV with mean/std/bootstrap CI columns.",
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
    for run_dir in sorted(experiments_root.rglob("run_*")):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "metadata.json").exists():
            continue
        row = aggregate_run(run_dir)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (row.get("dataset") or "", row.get("chunk_size") or 0, row.get("repeat_id") or 0))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if args.grouped_output_csv:
        grouped_rows = summarize_rows_by_group(rows)
        grouped_output_csv = Path(args.grouped_output_csv)
        if not grouped_output_csv.is_absolute():
            grouped_output_csv = repo_root / grouped_output_csv
        grouped_output_csv.parent.mkdir(parents=True, exist_ok=True)
        grouped_columns = summary_columns_for_grouped_rows(grouped_rows)
        with open(grouped_output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=grouped_columns)
            writer.writeheader()
            writer.writerows(grouped_rows)


def aggregate_run(run_dir: Path) -> dict | None:
    metadata = _read_json(run_dir / "metadata.json")
    stats = _read_json(run_dir / "stats.json")
    chunk_size = _parse_chunk_size(run_dir) or metadata.get("chunk_size")
    repeat_id = _parse_repeat_id(run_dir) or metadata.get("repeat_id")

    rule_path = _resolve_best_rule_path(run_dir, stats)
    train_eval = _read_eval_for_rule(rule_path, "train") or _best_eval(run_dir, "train")
    val_eval = _read_eval_for_rule(rule_path, "val") or _best_eval(run_dir, "val")
    test_eval = _read_eval_for_rule(rule_path, "test") or _best_eval(run_dir, "test")
    if not test_eval:
        return None
    rule_text = rule_path.read_text(encoding="utf-8") if rule_path and rule_path.exists() else ""
    densities = _trace_densities(run_dir)

    train_f1 = _metric_value(train_eval, "f1")
    val_f1 = _metric_value(val_eval, "f1")
    test_f1 = _metric_value(test_eval, "f1")
    event_test = _metric_block(test_eval, "event-based f1 under pa with mode squeeze")
    point_test = _metric_block(test_eval, "point-wise f1")
    point_fixed_test = _metric_block(test_eval, "point-wise fixed f1")
    point_pa_test = _metric_block(test_eval, "best f1 under pa") or _metric_block(
        test_eval, "point-wise f1 pa"
    )
    affiliation_test = _metric_block(test_eval, "affiliation f1")
    split_stats = metadata.get("split_stats") or stats.get("split_stats")

    row = {
        "dataset": metadata.get("dataset"),
        "chunk_size": chunk_size,
        "segment_selection_mode": metadata.get("segment_selection_mode"),
        "llm_provider": metadata.get("llm_provider"),
        "llm_engine": metadata.get("llm_engine") or metadata.get("resolved_llm_engine"),
        "temperature": metadata.get("temperature"),
        "top_k": metadata.get("top_k"),
        "max_iter": metadata.get("max_iter"),
        "seed": metadata.get("seed"),
        "repeat_id": repeat_id,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "precision": _metric_value(event_test or test_eval, "precision"),
        "recall": _metric_value(event_test or test_eval, "recall"),
        "point_f1": _metric_value(point_fixed_test or point_test, "f1"),
        "point_f1_fixed": _metric_value(point_fixed_test, "f1"),
        "point_f1_oracle": _metric_value(point_test, "f1"),
        "point_f1pa": _metric_value(point_pa_test, "f1"),
        "affiliation_f1": _metric_value(affiliation_test, "f1"),
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
    row.update(_flatten_split_stats(split_stats))
    return row


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


def summarize_rows_by_group(rows: list[dict], group_fields: list[str] | None = None) -> list[dict]:
    if not rows:
        return []
    if group_fields is None:
        group_fields = [
            "dataset",
            "segment_selection_mode",
            "llm_provider",
            "llm_engine",
            "temperature",
            "top_k",
            "max_iter",
            "seed",
            "chunk_size",
        ]

    metric_fields = [
        field
        for field in SUMMARY_COLUMNS
        if field not in group_fields and field not in {"repeat_id"}
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)

    grouped_rows = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        grouped_row = {field: value for field, value in zip(group_fields, key)}
        grouped_row["n_repeats"] = len(group_rows)
        for field in metric_fields:
            values = [row.get(field) for row in group_rows if _is_number(row.get(field))]
            if not values:
                continue
            grouped_row[f"{field}_mean"] = _mean(values)
            grouped_row[f"{field}_std"] = _std(values)
            ci_low, ci_high = _bootstrap_mean_ci(values)
            grouped_row[f"{field}_ci_low"] = ci_low
            grouped_row[f"{field}_ci_high"] = ci_high
        grouped_rows.append(grouped_row)
    return grouped_rows


def summary_columns_for_grouped_rows(grouped_rows: list[dict]) -> list[str]:
    if not grouped_rows:
        return ["n_repeats"]
    preferred = [
        "dataset",
        "segment_selection_mode",
        "llm_provider",
        "llm_engine",
        "temperature",
        "top_k",
        "max_iter",
        "seed",
        "chunk_size",
        "n_repeats",
    ]
    metric_suffixes = ["_mean", "_std", "_ci_low", "_ci_high"]
    seen = set()
    columns = []
    for field in preferred:
        if any(field in row for row in grouped_rows) and field not in seen:
            columns.append(field)
            seen.add(field)
    for row in grouped_rows:
        for field in row:
            if field in seen:
                continue
            if any(field.endswith(suffix) for suffix in metric_suffixes):
                continue
            columns.append(field)
            seen.add(field)
    metric_fields = sorted(
        {
            field[: -len(suffix)]
            for row in grouped_rows
            for field in row
            for suffix in metric_suffixes
            if field.endswith(suffix)
        }
    )
    for field in metric_fields:
        for suffix in metric_suffixes:
            columns.append(f"{field}{suffix}")
    return columns


def _bootstrap_mean_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05):
    if not values:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value

    rng = random.Random(42)
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(_mean(sample))
    means.sort()
    lower_idx = max(0, int((alpha / 2) * len(means)))
    upper_idx = min(len(means) - 1, int((1 - alpha / 2) * len(means)) - 1)
    return float(means[lower_idx]), float(means[upper_idx])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_split_stats(split_stats: dict | None) -> dict:
    if not isinstance(split_stats, dict):
        return {}

    expected_sections = ("train", "val", "test")
    if not all(section in split_stats for section in expected_sections):
        return {}

    flat = {}
    for section in expected_sections:
        section_stats = split_stats.get(section) or {}
        for key in (
            "total_points",
            "anomaly_point_count",
            "anomaly_event_count",
            "anomaly_point_ratio",
            "anomaly_event_ratio",
        ):
            flat[f"split_{section}_{key}"] = section_stats.get(key)

    flags = split_stats.get("flags") or {}
    flat["split_flags_test_event_count_lt_5"] = flags.get("test_event_count_lt_5")
    flat["split_flags_test_anomaly_point_count_lt_5"] = flags.get(
        "test_anomaly_point_count_lt_5"
    )
    return flat


def _parse_chunk_size(run_dir: Path):
    match = re.search(r"chunk_(\d+)", str(run_dir))
    return int(match.group(1)) if match else None


def _parse_repeat_id(run_dir: Path):
    match = re.search(r"run_(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    main()
