from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("reports/DECISION.md")


@dataclass(frozen=True)
class LegDecision:
    name: str
    status: str
    summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the final ARGOS phase-5 decision report.")
    parser.add_argument("--performance_csv", default="results/c1_comparison_suite_summary.csv")
    parser.add_argument("--cost_csv", default="results/chunk_sensitivity_summary.csv")
    parser.add_argument("--consistency_csv", default="results/c2_consistency_summary_phase1.csv")
    parser.add_argument("--morphology_csv", default="results/c3_morphology_diagnostic.csv")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--commit_hash", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    performance_rows = _load_csv(Path(args.performance_csv))
    cost_rows = _load_csv(Path(args.cost_csv))
    consistency_rows = _load_csv(Path(args.consistency_csv))
    morphology_rows = _load_csv(Path(args.morphology_csv))

    performance_leg = _evaluate_performance_leg(performance_rows)
    cost_leg = _evaluate_cost_leg(cost_rows)
    traceability_leg = _evaluate_traceability_leg(consistency_rows)
    conclusion = _decide_conclusion([performance_leg, cost_leg, traceability_leg])
    report = _render_report(
        performance_leg=performance_leg,
        cost_leg=cost_leg,
        traceability_leg=traceability_leg,
        conclusion=conclusion,
        performance_rows=performance_rows,
        cost_rows=cost_rows,
        consistency_rows=consistency_rows,
        morphology_rows=morphology_rows,
        commit_hash=args.commit_hash,
        performance_csv=args.performance_csv,
        cost_csv=args.cost_csv,
        consistency_csv=args.consistency_csv,
        morphology_csv=args.morphology_csv,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _evaluate_performance_leg(rows: list[dict[str, str]]) -> LegDecision:
    if not rows:
        return LegDecision("Performance", "WEAK", "No performance summary rows available.")

    evidence_rows = [row for row in rows if row.get("condition") in {"evidence", "event_bounded_reference"}]
    fixed_rows = [row for row in rows if row.get("condition") == "fixed"]
    if not evidence_rows or not fixed_rows:
        return LegDecision("Performance", "WEAK", "Missing fixed or evidence rows for the performance leg.")

    evidence_f1 = _mean(_extract_metric(row, ["mean_point_f1_fixed", "point_f1_fixed"]) for row in evidence_rows)
    fixed_f1 = _mean(_extract_metric(row, ["mean_point_f1_fixed", "point_f1_fixed"]) for row in fixed_rows)
    delta = evidence_f1 - fixed_f1

    if delta > 0.0:
        status = "PASS"
    elif delta < 0.0:
        status = "FAIL"
    else:
        status = "WEAK"

    return LegDecision(
        "Performance",
        status,
        f"evidence mean point_f1_fixed={_fmt(evidence_f1)} vs fixed mean point_f1_fixed={_fmt(fixed_f1)} (Δ={_fmt(delta)}).",
    )


def _evaluate_cost_leg(rows: list[dict[str, str]]) -> LegDecision:
    if not rows:
        return LegDecision("Cost", "WEAK", "No cost summary rows available.")

    evidence_rows = [row for row in rows if row.get("condition") == "evidence"]
    fixed_rows = [row for row in rows if row.get("condition") == "fixed"]
    if not evidence_rows or not fixed_rows:
        return LegDecision("Cost", "WEAK", "Missing fixed or evidence rows for the cost leg.")

    evidence_tokens = _mean(_extract_metric(row, ["mean_token_count_detection", "token_count_detection"]) for row in evidence_rows)
    fixed_tokens = _mean(_extract_metric(row, ["mean_token_count_detection", "token_count_detection"]) for row in fixed_rows)
    evidence_prompt_rows = _mean(_extract_metric(row, ["mean_prompt_rows", "prompt_rows"]) for row in evidence_rows)
    fixed_prompt_rows = _mean(_extract_metric(row, ["mean_prompt_rows", "prompt_rows"]) for row in fixed_rows)

    token_delta = fixed_tokens - evidence_tokens
    prompt_delta = fixed_prompt_rows - evidence_prompt_rows
    if token_delta > 0 and prompt_delta > 0:
        status = "PASS"
    elif token_delta < 0 and prompt_delta < 0:
        status = "FAIL"
    else:
        status = "WEAK"

    return LegDecision(
        "Cost",
        status,
        f"tokens fixed={_fmt(fixed_tokens)} evidence={_fmt(evidence_tokens)} (Δ={_fmt(token_delta)}); prompt rows fixed={_fmt(fixed_prompt_rows)} evidence={_fmt(evidence_prompt_rows)} (Δ={_fmt(prompt_delta)}).",
    )


def _evaluate_traceability_leg(rows: list[dict[str, str]]) -> LegDecision:
    if not rows:
        return LegDecision("Traceability", "WEAK", "No consistency summary rows available.")

    fixed_rows = [row for row in rows if row.get("condition") == "fixed"]
    evidence_rows = [row for row in rows if row.get("condition") == "evidence"]
    if not evidence_rows or not fixed_rows:
        return LegDecision("Traceability", "WEAK", "Missing fixed or evidence rows for the traceability leg.")

    evidence_gap = _mean(_extract_metric(row, ["heldout_support_gap_mean", "heldout_support_gap", "mean_heldout_support_gap"]) for row in evidence_rows)
    fixed_gap = _mean(_extract_metric(row, ["heldout_support_gap_mean", "heldout_support_gap", "mean_heldout_support_gap"]) for row in fixed_rows)
    delta = evidence_gap - fixed_gap

    if delta > 0:
        status = "PASS"
    elif delta < 0:
        status = "FAIL"
    else:
        status = "WEAK"

    return LegDecision(
        "Traceability",
        status,
        f"heldout_support_gap fixed={_fmt(fixed_gap)} evidence={_fmt(evidence_gap)} (Δ={_fmt(delta)}).",
    )


def _decide_conclusion(legs: list[LegDecision]) -> str:
    statuses = [leg.status for leg in legs]
    if statuses == ["PASS", "PASS", "PASS"]:
        return "GO"
    if statuses.count("FAIL") >= 1 and statuses.count("PASS") >= 1:
        return "GO-reframe"
    return "NO-GO"


def _render_report(
    *,
    performance_leg: LegDecision,
    cost_leg: LegDecision,
    traceability_leg: LegDecision,
    conclusion: str,
    performance_rows: list[dict[str, str]],
    cost_rows: list[dict[str, str]],
    consistency_rows: list[dict[str, str]],
    morphology_rows: list[dict[str, str]],
    commit_hash: str,
    performance_csv: str,
    cost_csv: str,
    consistency_csv: str,
    morphology_csv: str,
) -> str:
    lines: list[str] = []
    lines.append("# ARGOS Final Decision")
    lines.append("")
    lines.append(f"**Conclusion:** {conclusion}")
    lines.append("")
    lines.append("## Leg summary")
    for leg in [performance_leg, cost_leg, traceability_leg]:
        lines.append(f"- **{leg.name}**: {leg.status} — {leg.summary}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Performance CSV: `{performance_csv}` ({len(performance_rows)} rows)")
    lines.append(f"- Cost CSV: `{cost_csv}` ({len(cost_rows)} rows)")
    lines.append(f"- Consistency CSV: `{consistency_csv}` ({len(consistency_rows)} rows)")
    lines.append(f"- Morphology CSV: `{morphology_csv}` ({len(morphology_rows)} rows)")
    if commit_hash:
        lines.append(f"- Commit hash: `{commit_hash}`")
    lines.append("")
    lines.append("## Representative values")
    lines.extend(_representative_values(performance_rows, cost_rows, consistency_rows, morphology_rows))
    lines.append("")
    if conclusion != "GO":
        lines.append("## Recommended next step")
        lines.append("- Tighten the evidence condition around the series where the performance leg underperforms, then rerun the final comparison suite.")
        lines.append("")
    return "\n".join(lines)


def _representative_values(
    performance_rows: list[dict[str, str]],
    cost_rows: list[dict[str, str]],
    consistency_rows: list[dict[str, str]],
    morphology_rows: list[dict[str, str]],
) -> list[str]:
    lines: list[str] = []
    perf_evidence = [row for row in performance_rows if row.get("condition") in {"evidence", "event_bounded_reference"}]
    perf_fixed = [row for row in performance_rows if row.get("condition") == "fixed"]
    if perf_evidence and perf_fixed:
        lines.append(
            f"- Performance: evidence mean point_f1_fixed={_fmt(_mean(_to_float(row.get('mean_point_f1_fixed')) for row in perf_evidence))}, fixed mean point_f1_fixed={_fmt(_mean(_to_float(row.get('mean_point_f1_fixed')) for row in perf_fixed))}."
        )
    cost_evidence = [row for row in cost_rows if row.get("condition") == "evidence"]
    cost_fixed = [row for row in cost_rows if row.get("condition") == "fixed"]
    if cost_evidence and cost_fixed:
        lines.append(
            f"- Cost: evidence mean tokens={_fmt(_mean(_extract_metric(row, ['mean_token_count_detection', 'token_count_detection']) for row in cost_evidence))}, fixed mean tokens={_fmt(_mean(_extract_metric(row, ['mean_token_count_detection', 'token_count_detection']) for row in cost_fixed))}."
        )
    cons_evidence = [row for row in consistency_rows if row.get("condition") == "evidence"]
    cons_fixed = [row for row in consistency_rows if row.get("condition") == "fixed"]
    if cons_evidence and cons_fixed:
        lines.append(
            f"- Traceability: evidence mean heldout_support_gap={_fmt(_mean(_extract_metric(row, ['heldout_support_gap_mean', 'heldout_support_gap', 'mean_heldout_support_gap']) for row in cons_evidence))}, fixed mean heldout_support_gap={_fmt(_mean(_extract_metric(row, ['heldout_support_gap_mean', 'heldout_support_gap', 'mean_heldout_support_gap']) for row in cons_fixed))}."
        )
    if morphology_rows:
        label_counts: dict[str, int] = defaultdict(int)
        for row in morphology_rows:
            label_counts[row.get("morphology_label", "unknown")] += 1
        top_label = max(label_counts.items(), key=lambda item: item[1])[0]
        lines.append(f"- Morphology: most common label is `{top_label}` across {len(morphology_rows)} series.")
    return lines


def _mean(values: Any) -> float:
    vals = [v for v in values if _is_finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _extract_metric(row: dict[str, str], keys: list[str]) -> float:
    for key in keys:
        if key in row:
            value = _to_float(row.get(key))
            if _is_finite(value):
                return value
    return float("nan")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    if not _is_finite(value):
        return "nan"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
