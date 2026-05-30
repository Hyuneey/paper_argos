from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATASET_DIR = Path("results/datasets/kpi_preliminary")
DEFAULT_OUTPUT_CSV = Path("results/c3_morphology_diagnostic.csv")
DEFAULT_NOTE_PATH = Path("docs/reports/c3_morphology_diagnostic_note.md")


@dataclass(frozen=True)
class SeriesMorphology:
    dataset: str
    total_points: int
    anomaly_point_count: int
    anomaly_event_count: int
    anomaly_point_ratio: float
    anomaly_event_frequency_per_1k: float
    mean_event_len: float
    median_event_len: float
    max_event_len: float
    mean_event_amplitude: float
    max_event_amplitude: float
    mean_event_abs_slope: float
    max_event_abs_slope: float
    morphology_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a morphology diagnostic table for ARGOS series.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--comparison_rows_csv", default="results/c2_consistency_rows_phase1.csv")
    parser.add_argument("--performance_rows_csv", default="results/chunk_sensitivity_rows.csv")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--note_path", default=str(DEFAULT_NOTE_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    morphology_rows = _build_morphology_table(dataset_dir)
    performance_rows = _load_rows(Path(args.performance_rows_csv))
    consistency_rows = _load_rows(Path(args.comparison_rows_csv))

    perf_summary = _summarize_rows(performance_rows, key_fields=("dataset", "condition"))
    cons_summary = _summarize_rows(consistency_rows, key_fields=("dataset", "condition"))
    merged = _merge_diagnostics(morphology_rows, perf_summary, cons_summary)
    _write_csv(Path(args.output_csv), merged)
    _write_note(Path(args.note_path), merged)


def _build_morphology_table(dataset_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_path in sorted(dataset_dir.glob("*.csv")):
        df = pd.read_csv(dataset_path)
        rows.append(_series_morphology(df, dataset_path.stem).__dict__)
    return rows


def _series_morphology(df: pd.DataFrame, dataset: str) -> SeriesMorphology:
    labels = df["label"].to_numpy(dtype=int)
    values = df["value"].to_numpy(dtype=float)
    segments = _contiguous_anomaly_segments(labels)
    lengths = [end - start + 1 for start, end in segments]
    amplitudes = [float(np.max(values[start : end + 1]) - np.min(values[start : end + 1])) for start, end in segments] if segments else []
    abs_slopes = [float(np.mean(np.abs(np.diff(values[start : end + 1])))) if end > start else 0.0 for start, end in segments] if segments else []

    anomaly_point_count = int(np.sum(labels > 0.5))
    anomaly_event_count = len(segments)
    total_points = int(len(df))
    anomaly_point_ratio = float(anomaly_point_count / total_points) if total_points else 0.0
    anomaly_event_frequency = float(anomaly_event_count / total_points * 1000.0) if total_points else 0.0
    mean_event_len = float(np.mean(lengths)) if lengths else 0.0
    median_event_len = float(np.median(lengths)) if lengths else 0.0
    max_event_len = float(np.max(lengths)) if lengths else 0.0
    mean_event_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0
    max_event_amplitude = float(np.max(amplitudes)) if amplitudes else 0.0
    mean_event_abs_slope = float(np.mean(abs_slopes)) if abs_slopes else 0.0
    max_event_abs_slope = float(np.max(abs_slopes)) if abs_slopes else 0.0
    morphology_label = _classify_morphology(mean_event_len, mean_event_amplitude, mean_event_abs_slope, anomaly_event_count, anomaly_point_ratio)

    return SeriesMorphology(
        dataset=dataset,
        total_points=total_points,
        anomaly_point_count=anomaly_point_count,
        anomaly_event_count=anomaly_event_count,
        anomaly_point_ratio=anomaly_point_ratio,
        anomaly_event_frequency_per_1k=anomaly_event_frequency,
        mean_event_len=mean_event_len,
        median_event_len=median_event_len,
        max_event_len=max_event_len,
        mean_event_amplitude=mean_event_amplitude,
        max_event_amplitude=max_event_amplitude,
        mean_event_abs_slope=mean_event_abs_slope,
        max_event_abs_slope=max_event_abs_slope,
        morphology_label=morphology_label,
    )


def _classify_morphology(mean_event_len: float, mean_event_amplitude: float, mean_event_abs_slope: float, anomaly_event_count: int, anomaly_point_ratio: float) -> str:
    if anomaly_event_count == 0:
        return "no_anomaly"
    if mean_event_len <= 4 and mean_event_amplitude >= 0.02:
        return "spike"
    if mean_event_len >= 12 and mean_event_abs_slope <= 0.002:
        return "drift"
    if anomaly_point_ratio >= 0.12 and mean_event_len >= 8:
        return "plateau"
    return "mixed"


def _contiguous_anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(labels):
        if value > 0.5 and start is None:
            start = idx
        elif value <= 0.5 and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _summarize_rows(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        grouped[key].append(row)

    summary: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group_rows in grouped.items():
        summary[key] = {
            "rows": len(group_rows),
            **_mean_fields(group_rows, [
                "test_f1",
                "point_f1_fixed",
                "point_f1",
                "event_f1pa",
                "affiliation_f1",
                "runtime_sec",
                "token_count_detection",
                "prompt_rows",
                "heldout_support_gap",
                "local_support_gap",
                "heldout_anomaly_support_rate",
                "heldout_normal_violation_rate",
                "local_evidence_support_rate",
                "local_normal_reference_violation_rate",
            ]),
        }
    return summary


def _mean_fields(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in fields:
        values = [_to_float(row.get(field)) for row in rows if _is_float(row.get(field))]
        out[f"mean_{field}"] = float(np.mean(values)) if values else float("nan")
    return out


def _merge_diagnostics(
    morphology_rows: list[dict[str, Any]],
    perf_summary: dict[tuple[Any, ...], dict[str, Any]],
    cons_summary: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in morphology_rows:
        dataset = row["dataset"]
        fixed_perf = perf_summary.get((dataset, "fixed"), {})
        evidence_perf = perf_summary.get((dataset, "evidence"), {})
        fixed_cons = cons_summary.get((dataset, "fixed"), {})
        evidence_cons = cons_summary.get((dataset, "evidence"), {})
        merged.append(
            {
                **row,
                "fixed_rows": fixed_perf.get("rows", 0),
                "evidence_rows": evidence_perf.get("rows", 0),
                "delta_mean_test_f1": _delta(fixed_perf.get("mean_test_f1"), evidence_perf.get("mean_test_f1")),
                "delta_mean_point_f1_fixed": _delta(fixed_perf.get("mean_point_f1_fixed"), evidence_perf.get("mean_point_f1_fixed")),
                "delta_mean_point_f1": _delta(fixed_perf.get("mean_point_f1"), evidence_perf.get("mean_point_f1")),
                "delta_mean_event_f1pa": _delta(fixed_perf.get("mean_event_f1pa"), evidence_perf.get("mean_event_f1pa")),
                "delta_mean_token_count_detection": _delta(fixed_perf.get("mean_token_count_detection"), evidence_perf.get("mean_token_count_detection")),
                "delta_mean_prompt_rows": _delta(fixed_perf.get("mean_prompt_rows"), evidence_perf.get("mean_prompt_rows")),
                "fixed_mean_heldout_support_gap": fixed_cons.get("mean_heldout_support_gap"),
                "evidence_mean_heldout_support_gap": evidence_cons.get("mean_heldout_support_gap"),
                "delta_mean_heldout_support_gap": _delta(fixed_cons.get("mean_heldout_support_gap"), evidence_cons.get("mean_heldout_support_gap")),
                "fixed_mean_local_support_gap": fixed_cons.get("mean_local_support_gap"),
                "evidence_mean_local_support_gap": evidence_cons.get("mean_local_support_gap"),
                "delta_mean_local_support_gap": _delta(fixed_cons.get("mean_local_support_gap"), evidence_cons.get("mean_local_support_gap")),
            }
        )
    return merged


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _dedupe([key for row in rows for key in row.keys()])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_note(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# C3 Morphology Diagnostic Note",
        "",
        "This note summarizes the first-pass morphology table used for Phase 4.",
        "",
        "## Main pattern",
    ]
    if not rows:
        lines.append("- No rows were available.")
    else:
        finite_rows = []
        for row in rows:
            value = row.get("delta_mean_heldout_support_gap")
            if not _is_float(value):
                continue
            value_f = _to_float(value)
            if np.isnan(value_f):
                continue
            finite_rows.append((value_f, row))
        if finite_rows:
            rows_sorted = sorted(
                finite_rows,
                key=lambda item: (item[0], item[1].get("dataset", "")),
                reverse=True,
            )
            top_delta, top = rows_sorted[0]
            lines.append(
                f"- Highest finite support-gap delta: `{top.get('dataset')}` ({top.get('morphology_label')}), "
                f"Δheldout_support_gap={top_delta} and Δtest_f1={top.get('delta_mean_test_f1')}."
            )
        else:
            lines.append("- No paired fixed/evidence rows are available yet, so support-gap deltas remain undefined.")
        lines.append("- Use the CSV to inspect spike/drift/plateau rows against the fixed-vs-evidence deltas once phase2 completes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delta(a: Any, b: Any) -> float:
    if not (_is_float(a) and _is_float(b)):
        return float("nan")
    return float(a) - float(b)


def _to_float(value: Any) -> float:
    return float(value)


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


if __name__ == "__main__":
    main()
