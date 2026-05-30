from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_FIELDS = [
    "heldout_anomaly_support_rate",
    "heldout_normal_violation_rate",
    "heldout_support_gap",
    "local_evidence_support_rate",
    "local_normal_reference_violation_rate",
    "local_support_gap",
    "heldout_anomaly_window_count",
    "heldout_normal_window_count",
    "heldout_total_window_count",
]

PAIR_KEYS = ["dataset", "chunk_size", "repeat_id", "top_k", "max_iter", "seed"]


@dataclass(frozen=True)
class EvaluatedRun:
    row: dict[str, Any]
    run_dir: Path
    rule_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule evidence consistency on held-out windows.")
    parser.add_argument(
        "--experiments_roots",
        nargs="+",
        required=True,
        help="One or more experiment roots containing stats.json / metadata.json leaf runs.",
    )
    parser.add_argument(
        "--output_rows_csv",
        default="results/c2_consistency_rows.csv",
        help="Per-run row table.",
    )
    parser.add_argument(
        "--output_summary_csv",
        default="results/c2_consistency_summary.csv",
        help="Condition-level summary with paired fixed-vs-evidence delta.",
    )
    parser.add_argument(
        "--output_delta_csv",
        default="results/c2_consistency_delta.csv",
        help="Paired fixed-vs-evidence delta table.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=2000,
        help="Bootstrap samples used for CI estimation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _discover_run_dirs([Path(root) for root in args.experiments_roots])
    evaluated_runs = []
    for run_dir in run_dirs:
        result = _evaluate_run_directory(run_dir)
        if result is not None:
            evaluated_runs.append(result)

    row_records = [item.row for item in evaluated_runs]
    _write_csv(Path(args.output_rows_csv), row_records, row_fields=_row_fields(row_records))

    summary_rows = _summarize_by_condition(row_records, bootstrap_samples=args.bootstrap_samples)
    delta_rows = _paired_delta_rows(row_records, bootstrap_samples=args.bootstrap_samples)
    _write_csv(Path(args.output_summary_csv), summary_rows, row_fields=_row_fields(summary_rows))
    _write_csv(Path(args.output_delta_csv), delta_rows, row_fields=_row_fields(delta_rows))


def _discover_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: dict[Path, None] = {}
    for root in roots:
        if not root.exists():
            continue
        for stats_path in root.rglob("stats.json"):
            run_dirs[stats_path.parent.resolve()] = None
    return sorted(run_dirs.keys())


def _evaluate_run_directory(run_dir: Path) -> EvaluatedRun | None:
    metadata_path = run_dir / "metadata.json"
    stats_path = run_dir / "stats.json"
    if not metadata_path.exists() or not stats_path.exists():
        return None

    metadata = _load_json(metadata_path)
    stats = _load_json(stats_path)
    rule_path = _resolve_rule_path(run_dir, stats, metadata)
    if rule_path is None or not rule_path.exists():
        return None

    dataset_path = Path(metadata["dataset_path"])
    if not dataset_path.exists():
        return None

    df = pd.read_csv(dataset_path)
    held_out_pool = _load_held_out_pool(run_dir, stats, metadata)
    if not held_out_pool:
        return None

    rule_module = _load_rule_module(rule_path)
    local_metrics = _evaluate_local_trace(run_dir, rule_module, df, metadata, stats)
    heldout_metrics = _evaluate_heldout_pool(rule_module, df, held_out_pool)

    row = {
        "dataset": metadata.get("dataset", dataset_path.stem),
        "series": metadata.get("dataset", dataset_path.stem),
        "condition": metadata.get("segment_selection_mode", "fixed"),
        "llm_provider": metadata.get("llm_provider"),
        "llm_engine": metadata.get("resolved_llm_engine", metadata.get("llm_engine")),
        "temperature": metadata.get("temperature"),
        "top_k": metadata.get("top_k"),
        "max_iter": metadata.get("max_iter"),
        "seed": metadata.get("seed"),
        "repeat_id": metadata.get("repeat_id"),
        "chunk_size": metadata.get("chunk_size"),
        "run_dir": str(run_dir),
        "rule_path": str(rule_path),
        "held_out_window_pool_count": int(stats.get("held_out_window_pool_count", len(held_out_pool))),
        **heldout_metrics,
        **local_metrics,
        "split_test_total_points": _safe_nested_stat(metadata, "split_stats", "test", "total_points"),
        "split_test_anomaly_point_count": _safe_nested_stat(metadata, "split_stats", "test", "anomaly_point_count"),
        "split_test_anomaly_event_count": _safe_nested_stat(metadata, "split_stats", "test", "anomaly_event_count"),
    }
    return EvaluatedRun(row=row, run_dir=run_dir, rule_path=rule_path)


def _evaluate_heldout_pool(rule_module: Any, df: pd.DataFrame, pool: list[dict[str, Any]]) -> dict[str, Any]:
    anomaly_support: list[float] = []
    normal_violation: list[float] = []
    for item in pool:
        window_df = _window_slice(df, item)
        support = _window_support(rule_module, window_df)
        is_anomaly = int(item.get("anomaly_point_count", 0)) > 0 or int(item.get("anomaly_event_count", 0)) > 0
        if is_anomaly:
            anomaly_support.append(float(support))
        else:
            normal_violation.append(float(support))

    anomaly_rate = float(np.mean(anomaly_support)) if anomaly_support else float("nan")
    normal_rate = float(np.mean(normal_violation)) if normal_violation else float("nan")
    return {
        "heldout_anomaly_support_rate": anomaly_rate,
        "heldout_normal_violation_rate": normal_rate,
        "heldout_support_gap": anomaly_rate - normal_rate if _is_finite(anomaly_rate) and _is_finite(normal_rate) else float("nan"),
        "heldout_anomaly_window_count": len(anomaly_support),
        "heldout_normal_window_count": len(normal_violation),
        "heldout_total_window_count": len(pool),
    }


def _evaluate_local_trace(
    run_dir: Path,
    rule_module: Any,
    df: pd.DataFrame,
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    trace_path = _resolve_trace_path(run_dir, stats)
    if trace_path is None or not trace_path.exists():
        return {
            "local_evidence_support_rate": float("nan"),
            "local_normal_reference_violation_rate": float("nan"),
            "local_support_gap": float("nan"),
        }
    trace = _load_json(trace_path)
    provenance = trace.get("provenance", {})
    selected = trace.get("selected_segment", {})
    reference_segments = trace.get("reference_segments", {})
    selected_df = _slice_absolute_window(df, selected.get("start_pos"), selected.get("end_pos"))
    ref = reference_segments.get("normal_reference") if isinstance(reference_segments, dict) else None
    reference_df = _slice_absolute_window(df, ref.get("start_pos"), ref.get("end_pos")) if ref else None
    evidence_support = _window_support(rule_module, selected_df) if selected_df is not None else float("nan")
    reference_violation = _window_support(rule_module, reference_df) if reference_df is not None else float("nan")
    return {
        "local_evidence_support_rate": float(evidence_support),
        "local_normal_reference_violation_rate": float(reference_violation),
        "local_support_gap": float(evidence_support) - float(reference_violation)
        if _is_finite(evidence_support) and _is_finite(reference_violation)
        else float("nan"),
        "trace_path": str(trace_path),
        "selected_candidate_type": provenance.get("selected_candidate_type"),
    }


def _window_support(rule_module: Any, window_df: pd.DataFrame) -> bool:
    if window_df is None or window_df.empty:
        return False
    if "value" in window_df.columns:
        sample = window_df[["value"]].to_numpy(dtype=float)
    else:
        numeric = window_df.select_dtypes(include=[np.number])
        sample = numeric.to_numpy(dtype=float)
        if sample.size == 0:
            sample = window_df.to_numpy(dtype=float)
    labels = rule_module.inference(sample)
    return bool(np.asarray(labels).sum() > 0)


def _window_slice(df: pd.DataFrame, item: dict[str, Any]) -> pd.DataFrame:
    start = int(item["start_pos"])
    end = int(item["end_pos"])
    return df.iloc[start : end + 1].copy()


def _slice_absolute_window(df: pd.DataFrame, start_pos: Any, end_pos: Any) -> pd.DataFrame | None:
    if start_pos is None or end_pos is None:
        return None
    start = int(start_pos)
    end = int(end_pos)
    if end < start:
        return None
    return df.iloc[start : end + 1].copy()


def _resolve_rule_path(run_dir: Path, stats: dict[str, Any], metadata: dict[str, Any]) -> Path | None:
    rule_paths = stats.get("best_rule_paths") or []
    for raw_path in rule_paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            return path
        candidate = run_dir / raw_path
        if candidate.exists():
            return candidate
    meta_rule = metadata.get("rule_path")
    if meta_rule:
        path = Path(meta_rule)
        if path.exists():
            return path
        candidate = run_dir / meta_rule
        if candidate.exists():
            return candidate
    return None


def _resolve_trace_path(run_dir: Path, stats: dict[str, Any]) -> Path | None:
    trace_paths = stats.get("selection_trace_paths") or []
    for raw_path in trace_paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            return path
        candidate = run_dir / raw_path
        if candidate.exists():
            return candidate
    candidate = run_dir / "selection_trace_iter_0_call_0.json"
    if candidate.exists():
        return candidate
    traces = sorted(run_dir.glob("selection_trace_iter_*_call_*.json"))
    return traces[0] if traces else None


def _load_held_out_pool(run_dir: Path, stats: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    pool_path = stats.get("held_out_window_pool_path") or metadata.get("held_out_window_pool_path")
    if pool_path:
        path = Path(pool_path)
        if path.exists():
            payload = _load_json(path)
            return list(payload.get("pool", []))
        candidate = run_dir / pool_path
        if candidate.exists():
            payload = _load_json(candidate)
            return list(payload.get("pool", []))
    candidate = run_dir / "held_out_window_pool.json"
    if candidate.exists():
        payload = _load_json(candidate)
        return list(payload.get("pool", []))
    return []


def _load_rule_module(rule_path: Path):
    module_name = f"argos_rule_{abs(hash(rule_path)) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, rule_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load rule module from {rule_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "inference"):
        raise AttributeError(f"Rule module {rule_path} has no inference()")
    return module


def _summarize_by_condition(rows: list[dict[str, Any]], bootstrap_samples: int = 2000) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset"), row.get("condition"), row.get("chunk_size"))].append(row)

    summary_rows = []
    for (dataset, condition, chunk_size), group_rows in sorted(grouped.items()):
        summary_rows.append(
            _summary_row(
                group_rows,
                dataset=dataset,
                condition=condition,
                chunk_size=chunk_size,
                label="condition",
                bootstrap_samples=bootstrap_samples,
            )
        )
    return summary_rows


def _paired_delta_rows(rows: list[dict[str, Any]], bootstrap_samples: int = 2000) -> list[dict[str, Any]]:
    fixed_rows = [row for row in rows if row.get("condition") == "fixed"]
    evidence_rows = [row for row in rows if row.get("condition") == "evidence"]
    fixed_index = {tuple(row.get(key) for key in PAIR_KEYS): row for row in fixed_rows}
    evidence_index = {tuple(row.get(key) for key in PAIR_KEYS): row for row in evidence_rows}
    shared_keys = sorted(set(fixed_index).intersection(evidence_index))

    delta_rows: list[dict[str, Any]] = []
    for key in shared_keys:
        fixed = fixed_index[key]
        evidence = evidence_index[key]
        delta_rows.append(
            {
                "dataset": fixed.get("dataset"),
                "series": fixed.get("series"),
                "chunk_size": fixed.get("chunk_size"),
                "repeat_id": fixed.get("repeat_id"),
                "condition": "fixed_minus_evidence",
                "heldout_support_gap": _delta(fixed.get("heldout_support_gap"), evidence.get("heldout_support_gap")),
                "heldout_anomaly_support_rate": _delta(fixed.get("heldout_anomaly_support_rate"), evidence.get("heldout_anomaly_support_rate")),
                "heldout_normal_violation_rate": _delta(fixed.get("heldout_normal_violation_rate"), evidence.get("heldout_normal_violation_rate")),
                "local_support_gap": _delta(fixed.get("local_support_gap"), evidence.get("local_support_gap")),
                "pair_key": json.dumps(dict(zip(PAIR_KEYS, key)), ensure_ascii=False),
                "fixed_run_dir": fixed.get("run_dir"),
                "evidence_run_dir": evidence.get("run_dir"),
            }
        )

    if delta_rows:
        delta_rows.append(
            _summary_row(
                delta_rows,
                dataset="ALL",
                condition="fixed_minus_evidence",
                chunk_size=None,
                label="paired_delta",
                bootstrap_samples=bootstrap_samples,
            )
        )
    return delta_rows


def _summary_row(
    rows: list[dict[str, Any]],
    dataset: Any,
    condition: Any,
    chunk_size: Any,
    label: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset,
        "condition": condition,
        "chunk_size": chunk_size,
        "rows": len(rows),
        "summary_kind": label,
    }
    for field in METRIC_FIELDS:
        values = [_as_float(row.get(field)) for row in rows if _is_finite(row.get(field))]
        if not values:
            result[f"{field}_mean"] = float("nan")
            result[f"{field}_std"] = float("nan")
            result[f"{field}_ci_low"] = float("nan")
            result[f"{field}_ci_high"] = float("nan")
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci_low, ci_high = _bootstrap_ci(values, bootstrap_samples=bootstrap_samples)
        result[f"{field}_mean"] = mean
        result[f"{field}_std"] = std
        result[f"{field}_ci_low"] = ci_low
        result[f"{field}_ci_high"] = ci_high
    return result


def _bootstrap_ci(values: list[float], bootstrap_samples: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(max(1, bootstrap_samples)):
        boot = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(boot)))
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return lower, upper


def _delta(a: Any, b: Any) -> float:
    if not _is_finite(a) or not _is_finite(b):
        return float("nan")
    return float(a) - float(b)


def _safe_nested_stat(metadata: dict[str, Any], *keys: str) -> Any:
    current: Any = metadata
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]], row_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _dedupe(row_fields)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _row_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    return fields


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float:
    return float(value)


if __name__ == "__main__":
    main()
