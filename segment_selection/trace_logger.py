import json
import os
from datetime import datetime, timezone

from segment_selection.selector import SelectionResult


def write_selection_trace(
    selection_result: SelectionResult,
    trace_dir: str | None,
    iter_num: int,
    call_id: int,
    random_seed: int,
    config_path: str | None,
) -> str | None:
    if not trace_dir:
        return None
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"selection_trace_iter_{iter_num}_call_{call_id}.json")
    selected = selection_result.selected
    score = selection_result.selected_score
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "oracle_like_analysis_setting": True,
        "selected_segment": {
            "candidate_type": selected.kind,
            "start_pos": selected.start_pos,
            "end_pos": selected.end_pos,
            "start_index": selected.start_index,
            "end_index": selected.end_index,
            "length": selected.length,
        },
        "selection_score": {
            "total": score.total,
            **score.components,
        },
        "selection_rationale": list(selected.rationale),
        "reference_segments": {
            "normal_reference": _reference_payload(selected.reference_segment),
            "hard_negative": None,
        },
        "random_seed": random_seed,
        "selector_config_path": config_path,
        "selector_config_hash": selection_result.config_hash,
        "candidate_scores": [
            {
                "candidate_type": candidate.kind,
                "start_pos": candidate.start_pos,
                "end_pos": candidate.end_pos,
                "start_index": candidate.start_index,
                "end_index": candidate.end_index,
                "length": candidate.length,
                "score": {"total": candidate_score.total, **candidate_score.components},
            }
            for candidate, candidate_score in selection_result.candidate_scores
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return path


def _reference_payload(reference_segment):
    if reference_segment is None:
        return None
    return {"start_pos": reference_segment[0], "end_pos": reference_segment[1]}
