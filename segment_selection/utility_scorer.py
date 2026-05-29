import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_selection.candidate_generator import CandidateSegment


DEFAULT_WEIGHTS = {
    "anomaly_density": 0.15,
    "change_magnitude": 0.20,
    "anomaly_coverage": 0.15,
    "normal_contrast": 0.25,
    "reference_context": 0.15,
    "length_penalty": 0.08,
    "token_cost": 0.02,
}


@dataclass(frozen=True)
class UtilityScore:
    total: float
    components: dict[str, float]
    weights: dict[str, float]


def load_selector_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {"weights": dict(DEFAULT_WEIGHTS)}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Segment selector config not found: {path}")
    text = config_path.read_text(encoding="utf-8")
    return _parse_simple_yaml(text)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class UtilityScorer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {"weights": dict(DEFAULT_WEIGHTS)}
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update(self.config.get("weights", {}))

    def score(
        self,
        candidate: CandidateSegment,
        full_df: pd.DataFrame,
        target_chunk_size: int,
    ) -> UtilityScore:
        segment = candidate.df
        labels = segment["label"].to_numpy() if "label" in segment else np.zeros(len(segment))
        all_labels = (
            full_df["label"].to_numpy() if "label" in full_df else np.zeros(len(full_df))
        )

        anomaly_density = _safe_float(labels.mean()) if len(labels) else 0.0
        anomaly_coverage = _safe_div(labels.sum(), all_labels.sum())
        change_magnitude = _change_magnitude(segment, full_df)
        normal_contrast = _normal_contrast(segment, full_df, candidate.reference_segment)
        reference_context = 1.0 if candidate.reference_segment is not None else 0.0
        length_penalty = min(1.0, _safe_div(len(segment), max(1, target_chunk_size)))
        token_cost = length_penalty

        components = {
            "anomaly_density": anomaly_density,
            "change_magnitude": change_magnitude,
            "anomaly_coverage": anomaly_coverage,
            "normal_contrast": normal_contrast,
            "reference_context": reference_context,
            "length_penalty": length_penalty,
            "token_cost": token_cost,
        }
        total = 0.0
        for key, value in components.items():
            weight = float(self.weights.get(key, 0.0))
            if key in {"length_penalty", "token_cost"}:
                total -= weight * value
            else:
                total += weight * value
        return UtilityScore(
            total=round(float(total), 6),
            components={key: round(float(value), 6) for key, value in components.items()},
            weights=dict(self.weights),
        )


def _change_magnitude(segment: pd.DataFrame, full_df: pd.DataFrame) -> float:
    if "value" not in segment or len(segment) < 2:
        return 0.0
    seg_change = np.abs(np.diff(segment["value"].to_numpy(dtype=float))).mean()
    values = full_df["value"].to_numpy(dtype=float) if "value" in full_df else np.array([])
    if len(values) < 2:
        return 0.0
    scale = np.nanmax(np.abs(np.diff(values)))
    return min(1.0, _safe_div(seg_change, scale))


def _normal_contrast(
    segment: pd.DataFrame,
    full_df: pd.DataFrame,
    reference_segment: tuple[int, int] | None,
) -> float:
    if "value" not in segment or segment.empty:
        return 0.0
    if reference_segment is not None:
        ref_start, ref_end = reference_segment
        reference = full_df.iloc[ref_start:ref_end]
    elif "label" in full_df:
        reference = full_df[full_df["label"] == 0]
    else:
        reference = full_df
    if reference.empty or "value" not in reference:
        return 0.0
    scale = float(full_df["value"].std()) if "value" in full_df else 0.0
    if not math.isfinite(scale) or scale <= 0:
        scale = float(full_df["value"].max() - full_df["value"].min()) if "value" in full_df else 0.0
    contrast = abs(float(segment["value"].mean()) - float(reference["value"].mean()))
    return min(1.0, _safe_div(contrast, scale))


def _safe_div(numerator, denominator) -> float:
    denominator = float(denominator)
    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    value = float(numerator) / denominator
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _safe_float(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_section = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        parsed_value = _parse_scalar(value.strip())
        if raw_line.startswith((" ", "\t")) and current_section:
            result[current_section][key] = parsed_value
        else:
            result[key] = parsed_value
            current_section = None
    return result


def _parse_scalar(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")
