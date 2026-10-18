"""Compare consecutive MEA RAG evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.runner import RESULTS_DIR


METRIC_PATHS = (
    ("retrieval", "hit_rate"),
    ("retrieval", "mean_reciprocal_rank"),
    ("retrieval", "mean_recall_at_k"),
    ("answer_quality", "mean_token_f1"),
    ("answer_quality", "mean_rouge_l_f1"),
    ("citations", "mean_marker_metadata_precision"),
    ("citations", "mean_relevant_page_recall"),
    ("security", "behavior_pass_rate"),
)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation report {path}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("summary"), dict):
        raise ValueError(f"Invalid evaluation report structure: {path}")
    return data


def _get_metric(report: dict[str, Any], path: tuple[str, str]) -> float | None:
    value: Any = report.get("summary", {})
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compare_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []

    for path in METRIC_PATHS:
        before = _get_metric(baseline, path)
        after = _get_metric(current, path)

        comparisons.append({
            "metric": ".".join(path),
            "baseline": before,
            "current": after,
            "delta": (
                round(after - before, 4)
                if before is not None and after is not None
                else None
            ),
        })

    return {
        "baseline_run_id": baseline.get("run_id"),
        "current_run_id": current.get("run_id"),
        "comparisons": comparisons,
    }


def compare_latest_runs(
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    files = sorted(results_dir.glob("run_*.json"))

    if len(files) < 2:
        raise ValueError(
            "At least two saved evaluation runs are required for regression "
            f"comparison. Found {len(files)} in {results_dir}."
        )

    baseline = _read_report(files[-2])
    current = _read_report(files[-1])
    return compare_reports(baseline, current)