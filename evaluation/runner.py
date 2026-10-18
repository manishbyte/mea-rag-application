"""Run the MEA RAG evaluation dataset and persist actual results."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics.answer_quality import evaluate_answer_quality
from evaluation.metrics.citations import evaluate_citations
from evaluation.metrics.retrieval import evaluate_retrieval
from evaluation.metrics.security import evaluate_security
from graph.workflow import run_workflow

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "mea_v1.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


class EvaluationError(RuntimeError):
    """Raised when the dataset cannot be evaluated safely."""


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"Dataset file not found: {path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON on dataset line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise EvaluationError(
                    f"Dataset line {line_number} must be a JSON object."
                )

            for field in ("id", "question", "expected_behavior"):
                if not item.get(field):
                    raise EvaluationError(
                        f"Dataset line {line_number} is missing '{field}'."
                    )

            item_id = str(item["id"])
            if item_id in seen_ids:
                raise EvaluationError(f"Duplicate dataset ID: {item_id}")
            seen_ids.add(item_id)

            behavior = str(item["expected_behavior"]).lower()
            if behavior not in {"answer", "abstain", "reject"}:
                raise EvaluationError(
                    f"Unsupported expected_behavior '{behavior}' "
                    f"for dataset item {item_id}."
                )

            if not isinstance(item.get("question"), str):
                raise EvaluationError(
                    f"Question must be text for dataset item {item_id}."
                )

            records.append(item)

    if not records:
        raise EvaluationError("The evaluation dataset is empty.")

    return records


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [item["metrics"]["retrieval"] for item in results]
    quality = [item["metrics"]["answer_quality"] for item in results]
    citations = [item["metrics"]["citations"] for item in results]
    security = [item["metrics"]["security"] for item in results]

    def numeric_metric(items: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(item[key])
            for item in items
            if isinstance(item.get(key), (int, float))
            and not isinstance(item.get(key), bool)
        ]

    latencies = [
        float(item["latency_seconds"])
        for item in results
        if isinstance(item.get("latency_seconds"), (int, float))
    ]

    behavior_counts = Counter(
        item["metrics"]["security"]["expected_behavior"] for item in results
    )

    return {
        "total_questions": len(results),
        "successful_executions": sum(item["execution_ok"] for item in results),
        "failed_executions": sum(not item["execution_ok"] for item in results),
        "retrieval": {
            "hit_rate": _rate([
                item["hit"] for item in retrieval
                if isinstance(item.get("hit"), bool)
            ]),
            "mean_reciprocal_rank": _mean(
                numeric_metric(retrieval, "reciprocal_rank")
            ),
            "mean_recall_at_k": _mean(
                numeric_metric(retrieval, "recall_at_k")
            ),
            "mean_page_recall": _mean(
                numeric_metric(retrieval, "page_recall")
            ),
        },
        "answer_quality": {
            "mean_token_f1": _mean(
                numeric_metric(quality, "token_f1")
            ),
            "mean_rouge_l_f1": _mean(
                numeric_metric(quality, "rouge_l_f1")
            ),
            "exact_match_rate": _rate([
                item["exact_match"] for item in quality
                if isinstance(item.get("exact_match"), bool)
            ]),
        },
        "citations": {
            "mean_marker_metadata_precision": _mean(
                numeric_metric(citations, "citation_marker_metadata_precision")
            ),
            "mean_relevant_page_recall": _mean(
                numeric_metric(citations, "relevant_page_citation_recall")
            ),
        },
        "security": {
            "behavior_pass_rate": _rate([
                item["behavior_passed"] for item in security
                if isinstance(item.get("behavior_passed"), bool)
            ]),
            "expected_behavior_counts": dict(behavior_counts),
        },
        "latency": {
            "mean_seconds": _mean(latencies),
            "max_seconds": round(max(latencies), 4) if latencies else None,
        },
    }


async def _evaluate_one(item: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "id": item["id"],
        "question": item["question"],
        "expected_behavior": item["expected_behavior"],
        "security_category": item.get("security_category"),
        "execution_ok": False,
        "latency_seconds": None,
        "answer": "",
        "citations": [],
        "abstained": False,
        "workflow_error": None,
        "metrics": {},
    }

    try:
        state = await run_workflow(item["question"])
        answer = str(state.get("answer") or "")
        citations = state.get("citations") or []
        chunks = state.get("retrieved_chunks") or []
        abstained = bool(state.get("abstained", False))

        result.update({
            "execution_ok": True,
            "answer": answer,
            "citations": citations,
            "abstained": abstained,
            "validation_passed": state.get("validation_passed"),
            "validation_errors": state.get("validation_errors", []),
            "workflow_error": state.get("workflow_error"),
            "retrieval_error": state.get("retrieval_error"),
        })

        result["metrics"] = {
            "retrieval": evaluate_retrieval(
                chunks,
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                relevant_pages=item.get("relevant_pages", []),
                k=5,
            ),
            "answer_quality": evaluate_answer_quality(
                answer, item.get("expected_answer")
            ),
            "citations": evaluate_citations(
                answer,
                citations,
                chunks,
                relevant_pages=item.get("relevant_pages", []),
            ),
            "security": evaluate_security(
                expected_behavior=item["expected_behavior"],
                answer=answer,
                abstained=abstained,
                validation_passed=state.get("validation_passed"),
            ),
        }

    except Exception as exc:
        # Do not write exception details into user-facing answers.
        # Logs keep the traceback for operator debugging.
        logger.exception("Evaluation item %s failed", item["id"])
        result["workflow_error"] = type(exc).__name__
        result["metrics"] = {
            "retrieval": evaluate_retrieval(
                [],
                item.get("relevant_chunk_ids", []),
                item.get("relevant_pages", []),
            ),
            "answer_quality": evaluate_answer_quality(
                "", item.get("expected_answer")
            ),
            "citations": evaluate_citations(
                "", [], [], item.get("relevant_pages", [])
            ),
            "security": evaluate_security(
                item["expected_behavior"], "", False, None
            ),
        }

    finally:
        result["latency_seconds"] = round(
            time.perf_counter() - started, 4
        )

    return result


async def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)

    # Temporarily evaluate only the first 3 questions.
    dataset = dataset[:2]

    results: list[dict[str, Any]] = []

    for index, item in enumerate(dataset, start=1):
        logger.info(
            "Evaluating %s (%d/%d)", item["id"], index, len(dataset)
        )
        results.append(await _evaluate_one(item))

    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        "dataset_version": dataset[0].get("dataset_version", "unknown"),
        "summary": aggregate_results(results),
        "results": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"run_{report['run_id']}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["result_file"] = str(output_path)
    logger.info("Evaluation results saved to %s", output_path)
    return report