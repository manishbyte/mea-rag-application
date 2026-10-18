
"""Atomic JSONL export for processed RAG document chunks."""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID


DEFAULT_OUTPUT_PATH = Path(
    "data/processed/mea_annual_report_chunks.jsonl"
)


class ExportError(RuntimeError):
    """Raised when chunk export fails."""


def _json_default(value: Any) -> Any:
    """Convert supported non-JSON-native values into JSON values."""
    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def _validate_record(record: Mapping[str, Any], index: int) -> None:
    """Validate the minimum metadata required for a dataset record."""
    required_fields = (
        "chunk_index",
        "page_number",
        "content",
    )

    missing = [
        field
        for field in required_fields
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"Record {index} is missing required fields: "
            f"{', '.join(missing)}."
        )

    if not isinstance(record["chunk_index"], int) or isinstance(
        record["chunk_index"], bool
    ):
        raise ValueError(
            f"Record {index} has an invalid chunk_index."
        )

    if record["chunk_index"] < 0:
        raise ValueError(
            f"Record {index} has a negative chunk_index."
        )

    if not isinstance(record["page_number"], int) or isinstance(
        record["page_number"], bool
    ):
        raise ValueError(
            f"Record {index} has an invalid page_number."
        )

    if record["page_number"] < 1:
        raise ValueError(
            f"Record {index} has an invalid page_number."
        )

    if not isinstance(record["content"], str):
        raise ValueError(
            f"Record {index} content must be a string."
        )

    if not record["content"].strip():
        raise ValueError(
            f"Record {index} content must not be empty."
        )


def export_chunks_jsonl(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Atomically export chunk records as UTF-8 JSON Lines.

    Each input mapping becomes one JSON object on one line.
    The destination is replaced only after the complete temporary
    file has been written successfully.

    Returns:
        The resolved path of the exported dataset.

    Raises:
        ValueError: If a record is invalid.
        ExportError: If writing or replacing the output fails.
    """
    destination = Path(output_path).expanduser().resolve()

    # Validate everything before creating or replacing the destination.
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Record {index} must be a mapping."
            )

        _validate_record(record, index)

        try:
            # Detect unsupported values before starting the write.
            json.dumps(
                dict(record),
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record {index} cannot be serialized to JSON."
            ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            for record in records:
                line = json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
                temporary_file.write(line + "\n")

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # The temporary file is in the destination directory so the
        # replacement stays on the same filesystem.
        os.replace(temporary_path, destination)
        temporary_path = None

        return destination

    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(
            f"Could not export chunks to {destination}."
        ) from exc

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
