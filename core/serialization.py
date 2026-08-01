"""JSON serialization helpers for algorithm results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class SupportsToDict(Protocol):
    """Protocol implemented by result contracts."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""


def save_result_json(result: SupportsToDict, path: str | Path) -> Path:
    """Write a result as indented UTF-8 JSON and return the output path.

    Args:
        result: Search or multi-location result implementing ``to_dict``.
        path: Destination file path. Parent folders are created when needed.

    Returns:
        The normalized ``Path`` used for the output.

    Raises:
        TypeError: If ``result`` does not provide a callable ``to_dict`` method.
        OSError: If the destination cannot be written.
    """
    if not callable(getattr(result, "to_dict", None)):
        raise TypeError("result must provide a callable to_dict() method")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return destination
