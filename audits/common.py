"""Common helpers for prospective audit scripts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_field_filled(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def source_hash_matches(
    record: dict[str, Any],
    *,
    path_key: str,
    hash_key: str,
    base_dir: str | Path,
) -> bool:
    declared_hash = record.get(hash_key)
    declared_path = record.get(path_key)
    if not hash_field_filled(declared_hash) or not isinstance(declared_path, str) or not declared_path:
        return False
    source_path = Path(declared_path)
    if not source_path.is_absolute():
        source_path = Path(base_dir) / source_path
    if not source_path.is_file():
        return False
    return sha256_file(source_path) == declared_hash.lower()
