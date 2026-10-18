"""CLI entry point for the MEA RAG evaluation suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from evaluation.regression import compare_latest_runs
from evaluation.runner import DEFAULT_DATASET, RESULTS_DIR, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MEA RAG chatbot against a JSONL dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where JSON evaluation reports are saved.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the latest two saved runs instead of running evaluation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    if args.compare:
        report = compare_latest_runs(args.results_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    report = await run_evaluation(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
    )

    print("\n=== MEA RAG Evaluation Summary ===")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nDetailed results: {report['result_file']}")

    failed = report["summary"]["failed_executions"]
    security_rate = report["summary"]["security"]["behavior_pass_rate"]

    if failed:
        return 1
    if security_rate is not None and security_rate < 1.0:
        print(
            "\nWARNING: At least one expected-behavior security/abstention "
            "check failed. Review the per-question results."
        )
        return 2

    return 0


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        exit_code = asyncio.run(_main(args))
    except (ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()