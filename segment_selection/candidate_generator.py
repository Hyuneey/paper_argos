from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSegment:
    kind: str
    df: pd.DataFrame
    start_pos: int
    end_pos: int
    rationale: tuple[str, ...]
    reference_segment: tuple[int, int] | None = None

    @property
    def length(self) -> int:
        return max(0, self.end_pos - self.start_pos)

    @property
    def start_index(self):
        if self.df.empty or "index" not in self.df:
            return None
        return _json_number(self.df["index"].iloc[0])

    @property
    def end_index(self):
        if self.df.empty or "index" not in self.df:
            return None
        return _json_number(self.df["index"].iloc[-1])


def generate_candidates(
    df: pd.DataFrame,
    chunk_size: int,
    iter_num: int,
    fixed_df: pd.DataFrame | None = None,
    random_seed: int = 8,
) -> list[CandidateSegment]:
    """Generate label-based oracle candidates for research analysis.

    This is intentionally an oracle-like analysis setting: candidate placement may
    use ground-truth labels. The generated rule still receives labels only through
    the same prompt channel as ARGOS's fixed chunks.
    """
    if df.empty:
        return []

    candidates = []
    fixed_df = fixed_df if fixed_df is not None else _window(df, 0, min(chunk_size, len(df)))
    fixed_start, fixed_end = _position_bounds(df, fixed_df)
    candidates.append(
        CandidateSegment(
            kind="fixed_chunk",
            df=fixed_df.copy(),
            start_pos=fixed_start,
            end_pos=fixed_end,
            rationale=("baseline fixed chunk selected by ARGOS iteration sampling",),
        )
    )

    anomaly_segments = _anomaly_segments(df)
    total_len = len(df)
    rng = np.random.default_rng(random_seed + int(iter_num))

    for start, end in anomaly_segments:
        center = (start + end) // 2
        for kind, fraction in (
            ("short_anomaly_centered", 0.25),
            ("medium_anomaly_centered", 0.50),
            ("long_anomaly_centered", 1.00),
        ):
            length = max(1, min(total_len, int(round(chunk_size * fraction))))
            segment_start, segment_end = _centered_bounds(center, length, total_len)
            candidates.append(
                CandidateSegment(
                    kind=kind,
                    df=_window(df, segment_start, segment_end),
                    start_pos=segment_start,
                    end_pos=segment_end,
                    rationale=(
                        f"window centered near labeled anomaly segment {start}:{end}",
                        "oracle-like analysis setting using labels for segment selection",
                    ),
                )
            )

        normal_ref = _nearest_normal_window(df, start, max(1, min(chunk_size, end - start)))
        if normal_ref is not None:
            ref_start, ref_end = normal_ref
            candidates.append(
                CandidateSegment(
                    kind="nearby_normal_reference",
                    df=_window(df, ref_start, ref_end),
                    start_pos=ref_start,
                    end_pos=ref_end,
                    rationale=("nearby all-normal segment for contrast",),
                    reference_segment=normal_ref,
                )
            )

    if total_len > 0:
        random_len = min(chunk_size, total_len)
        random_start = int(rng.integers(0, max(1, total_len - random_len + 1)))
        random_end = random_start + random_len
        candidates.append(
            CandidateSegment(
                kind="random_segment",
                df=_window(df, random_start, random_end),
                start_pos=random_start,
                end_pos=random_end,
                rationale=("random segment used as selection ablation",),
            )
        )

    return _dedupe_candidates(candidates)


def _anomaly_segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    labels = df["label"].to_numpy() if "label" in df else np.zeros(len(df))
    segments = []
    start = None
    for idx, label in enumerate(labels):
        if label == 1 and start is None:
            start = idx
        elif label != 1 and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def _nearest_normal_window(
    df: pd.DataFrame, anchor: int, length: int
) -> tuple[int, int] | None:
    labels = df["label"].to_numpy() if "label" in df else np.zeros(len(df))
    if len(labels) < length:
        return None
    best = None
    best_distance = float("inf")
    for start in range(0, len(labels) - length + 1):
        end = start + length
        if labels[start:end].sum() != 0:
            continue
        distance = abs(start - anchor)
        if distance < best_distance:
            best = (start, end)
            best_distance = distance
    return best


def _centered_bounds(center: int, length: int, total_len: int) -> tuple[int, int]:
    start = max(0, center - length // 2)
    end = min(total_len, start + length)
    start = max(0, end - length)
    return start, end


def _window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df.iloc[start:end].copy()


def _position_bounds(df: pd.DataFrame, segment_df: pd.DataFrame) -> tuple[int, int]:
    if segment_df.empty:
        return 0, 0
    first_label = segment_df.index[0]
    start_positions = np.flatnonzero(df.index.to_numpy() == first_label)
    start = int(start_positions[0]) if len(start_positions) else 0
    return start, start + len(segment_df)


def _dedupe_candidates(candidates: list[CandidateSegment]) -> list[CandidateSegment]:
    seen = set()
    deduped = []
    for candidate in candidates:
        key = (candidate.kind, candidate.start_pos, candidate.end_pos)
        if key in seen or candidate.df.empty:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _json_number(value):
    if hasattr(value, "item"):
        return value.item()
    return value
