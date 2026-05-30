from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = [
    "chunk_size",
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


SUMMARY_FIELDS = [
    "condition",
    "dataset",
    "rows",
    "mean_val_f1",
    "mean_test_f1",
    "mean_point_f1_fixed",
    "mean_runtime_sec",
    "mean_token_count_detection",
    "mean_prompt_rows",
    "std_val_f1",
    "std_test_f1",
    "std_point_f1_fixed",
    "ci95_val_f1",
    "ci95_test_f1",
    "ci95_point_f1_fixed",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate comparison-suite outputs.")
    parser.add_argument(
        "--suite_root",
        default="experiments/c1_comparison_suite",
        help="Root directory produced by run_comparison_suite.py.",
    )
    parser.add_argument(
        "--output_rows_csv",
        default="results/c1_comparison_suite_rows.csv",
        help="Combined per-run rows across all conditions.",
    )
    parser.add_argument(
        "--output_summary_csv",
        default="results/c1_comparison_suite_summary.csv",
        help="Condition-level summary across runs.",
    )
    parser.add_argument(
        "--output_selected_csv",
        default="results/c1_comparison_suite_best_fixed_selected.csv",
        help="Series-level best-fixed(val) selected rows.",
    )
    parser.add_argument(
        "--aggregate_script",
        default="scripts/aggregate_chunk_sensitivity.py",
        help="Path to the existing chunk sensitivity aggregator.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = Path(args.suite_root)
    if not suite_root.is_absolute():
        suite_root = repo_root / suite_root
    if not suite_root.exists():
        raise FileNotFoundError(f"Suite root not found: {suite_root}")

    aggregate_script = Path(args.aggregate_script)
    if not aggregate_script.is_absolute():
        aggregate_script = repo_root / aggregate_script

    leaf_roots = sorted(
        path
        for path in suite_root.glob("*/*")
        if path.is_dir() and list(path.glob("chunk_*/run_*"))
    )
    if not leaf_roots:
        raise FileNotFoundError(f"No condition leaf roots found under {suite_root}")

    condition_rows = []
    for leaf_root in leaf_roots:
        condition = leaf_root.parent.name
        series = leaf_root.name
        leaf_summary = leaf_root / "comparison_summary.csv"
        subprocess.run(
            [
                sys.executable,
                str(aggregate_script),
                "--experiments_root",
                str(leaf_root),
                "--output_csv",
                str(leaf_summary),
            ],
            cwd=repo_root,
            check=True,
        )
        condition_rows.extend(
            _load_rows(leaf_summary, condition=condition, series=series, suite_root=suite_root)
        )

    output_rows_csv = Path(args.output_rows_csv)
    if not output_rows_csv.is_absolute():
        output_rows_csv = repo_root / output_rows_csv
    output_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_rows_csv, condition_rows, row_fields=["condition", "series"] + _all_fields(condition_rows))

    summary_rows = _summarize_conditions(condition_rows)
    output_summary_csv = Path(args.output_summary_csv)
    if not output_summary_csv.is_absolute():
        output_summary_csv = repo_root / output_summary_csv
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_summary_csv, summary_rows, row_fields=SUMMARY_FIELDS)

    selected_rows = _best_fixed_selected_rows(condition_rows)
    output_selected_csv = Path(args.output_selected_csv)
    if not output_selected_csv.is_absolute():
        output_selected_csv = repo_root / output_selected_csv
    output_selected_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_selected_csv, selected_rows, row_fields=_all_fields(selected_rows))


def _load_rows(csv_path: Path, condition: str, series: str, suite_root: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = dict(row)
            normalized["condition"] = condition
            normalized["series"] = series
            normalized["suite_root"] = str(suite_root)
            rows.append(normalized)
    return rows


def _all_fields(rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    preferred = [
        "suite_root",
        "condition",
        "series",
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
    seen: set[str] = set()
    for field in preferred:
        if any(field in row for row in rows) and field not in seen:
            ordered.append(field)
            seen.add(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                ordered.append(field)
                seen.add(field)
    return ordered


def _write_csv(path: Path, rows: list[dict], row_fields: list[str]) -> None:
    deduped_fields = list(dict.fromkeys(row_fields))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=deduped_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_conditions(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("condition"), row.get("dataset"))].append(row)
    summary_rows = []
    for (condition, dataset), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(_summary_row(condition, dataset, group_rows))
    return summary_rows


def _summary_row(condition: str, dataset: str, rows: list[dict]) -> dict:
    vals = {field: _numeric_list(rows, field) for field in ("val_f1", "test_f1", "point_f1_fixed", "runtime_sec", "token_count_detection", "prompt_rows")}
    summary = {
        "condition": condition,
        "dataset": dataset,
        "rows": len(rows),
        "mean_val_f1": _mean(vals["val_f1"]),
        "mean_test_f1": _mean(vals["test_f1"]),
        "mean_point_f1_fixed": _mean(vals["point_f1_fixed"]),
        "mean_runtime_sec": _mean(vals["runtime_sec"]),
        "mean_token_count_detection": _mean(vals["token_count_detection"]),
        "mean_prompt_rows": _mean(vals["prompt_rows"]),
        "std_val_f1": _std(vals["val_f1"]),
        "std_test_f1": _std(vals["test_f1"]),
        "std_point_f1_fixed": _std(vals["point_f1_fixed"]),
        "ci95_val_f1": _ci95(vals["val_f1"]),
        "ci95_test_f1": _ci95(vals["test_f1"]),
        "ci95_point_f1_fixed": _ci95(vals["point_f1_fixed"]),
    }
    return summary


def _best_fixed_selected_rows(rows: list[dict]) -> list[dict]:
    fixed_rows = [row for row in rows if row.get("condition") == "fixed"]
    grouped = defaultdict(list)
    for row in fixed_rows:
        grouped[row.get("series")].append(row)
    selected_rows = []
    for series, series_rows in sorted(grouped.items(), key=lambda item: item[0]):
        by_chunk = defaultdict(list)
        for row in series_rows:
            by_chunk[row.get("chunk_size")].append(row)
        best_chunk = None
        best_val = None
        for chunk_size, chunk_rows in by_chunk.items():
            val_f1 = _mean(_numeric_list(chunk_rows, "val_f1"))
            if best_val is None or (val_f1 is not None and val_f1 > best_val):
                best_val = val_f1
                best_chunk = chunk_size
        if best_chunk is None:
            continue
        chosen_rows = by_chunk[best_chunk]
        selected_rows.append(
            _selected_summary_row(series, best_chunk, chosen_rows, best_val)
        )
    return selected_rows


def _selected_summary_row(series: str, chunk_size: str | int, rows: list[dict], best_val: float | None) -> dict:
    base = dict(rows[0])
    base["condition"] = "best_fixed_val"
    base["series"] = series
    base["chunk_size"] = chunk_size
    base["selection_key"] = "mean_val_f1"
    base["selection_value"] = best_val
    for field in NUMERIC_FIELDS:
        if field in base:
            values = _numeric_list(rows, field)
            if values:
                base[field] = _mean(values)
    base["rows"] = len(rows)
    return base


def _numeric_list(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, "", "None"):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def _ci95(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


if __name__ == "__main__":
    main()
