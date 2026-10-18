
"""Local entry point for testing the complete MEA RAG workflow."""

from __future__ import annotations

import asyncio
import json

from graph.workflow import run_workflow


async def main() -> None:
    print("=" * 70)
    print("MEA ANNUAL REPORT 2025 — RAG CHATBOT")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("\nError: Question cannot be empty.")
        return

    print("\nRunning the RAG workflow...\n")

    try:
        result = await run_workflow(question)

        print("=" * 70)
        print("FULL WORKFLOW OUTPUT")
        print("=" * 70)

        print("\nQUESTION")
        print("-" * 70)
        print(result.get("question", question))

        print("\nANSWER")
        print("-" * 70)
        print(result.get("answer", "No answer returned."))

        print("\nCITATIONS")
        print("-" * 70)
        print(
            json.dumps(
                result.get("citations", []),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\nEVIDENCE STATUS")
        print("-" * 70)
        print(f"Sufficient: {result.get('evidence_sufficient', False)}")
        print(f"Reason: {result.get('evidence_reason', '')}")
        print(f"Abstained: {result.get('abstained', False)}")

        print("\nOUTPUT VALIDATION")
        print("-" * 70)
        print(f"Passed: {result.get('validation_passed', False)}")
        print(
            json.dumps(
                result.get("validation_errors", []),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nRETRIEVAL")
        print("-" * 70)
        chunks = result.get("retrieved_chunks", [])
        print(f"Retrieved chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {index} ---")
            print(f"Document: {chunk.get('document_title', 'Unknown')}")
            print(f"Page: {chunk.get('page_number', 'Unknown')}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}")
            print(f"Method: {chunk.get('retrieval_method', 'Unknown')}")
            print("Content:")
            print(chunk.get("content", ""))

        print("\nWORKFLOW ERRORS")
        print("-" * 70)
        print(f"Retrieval error: {result.get('retrieval_error', '')}")
        print(f"Workflow error: {result.get('workflow_error', '')}")

        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETED")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nExecution cancelled.")

    except Exception as exc:
        # Avoid printing provider response bodies or secrets in production.
        print(f"\nWorkflow failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
