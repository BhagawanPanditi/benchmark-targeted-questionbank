"""JSON persistence helpers: atomic saves, resumability, input validation."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_json(path: str | Path, data: Any) -> None:
    """Atomically write *data* as JSON: write to path + '.tmp', then os.replace.

    Saves are full-file: every stage calls this after each processed record so a
    crash never loses more than the in-flight record.
    """
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def load_json(path: str | Path) -> list[dict]:
    """Load a JSON array of record objects.

    Returns [] when the file is missing, unreadable, or not a list of objects
    (a corrupt intermediate file is treated as "no completed work yet" so the
    stage simply regenerates it; the corruption is logged loudly).
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("failed to load %s: %s", path, exc)
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    logger.error("%s does not contain a JSON array of objects", path)
    return []


def load_json_obj(path: str | Path) -> Any | None:
    """Load any JSON document (dict or list). None when missing or invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("failed to load %s: %s", path, exc)
        return None


def load_existing_ids(path: str | Path, id_field: str) -> set[str]:
    """Collect the values of *id_field* from an existing output file.

    Used by every stage for resumability: records whose id is already present
    are skipped on re-run.
    """
    ids: set[str] = set()
    for record in load_json(path):
        value = record.get(id_field)
        if value is not None:
            ids.add(str(value))
    return ids


def require_file(path: str | Path, hint: str = "") -> Path:
    """Fail fast with a clear message when a required upstream file is absent."""
    path = Path(path)
    if not path.exists():
        message = f"error: required input file not found: {path}"
        if hint:
            message += f" {hint}"
        raise SystemExit(message)
    return path
