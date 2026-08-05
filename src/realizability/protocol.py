"""Frozen protocol helpers for path-level realizability audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class RealizabilityProtocolError(ValueError):
    """Raised when a realizability protocol is incomplete or modified."""


REQUIRED_PROTOCOL_FIELDS = {
    "parameter_interval",
    "derivative_method",
    "integration_method",
    "parameter_grid_points",
    "path_nonconstant_tol",
    "path_displacement_nonconstant_tol",
    "velocity_nonzero_tol",
    "local_cost_zero_tol",
    "total_cost_zero_tol",
    "parameter_grid_absolute_tol",
    "parameter_grid_relative_tol",
    "velocity_derivative_absolute_tol",
    "velocity_derivative_relative_tol",
    "summary_accumulated_cost_absolute_tol",
    "summary_positive_measure_absolute_tol",
    "summary_control_cost_absolute_tol",
    "minimum_positive_measure_fraction",
    "minimum_positive_control_cost",
    "contraction_sequence_acceptance_rule",
    "protocol_sha256",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def protocol_sha256(protocol: dict[str, Any]) -> str:
    comparable = dict(protocol)
    comparable.pop("protocol_sha256", None)
    return canonical_hash(comparable)


def require_matching_protocol_sha(protocol: dict[str, Any]) -> str:
    missing = sorted(REQUIRED_PROTOCOL_FIELDS - set(protocol))
    if missing:
        raise RealizabilityProtocolError(
            "realizability protocol missing fields: " + ", ".join(missing)
        )
    declared = protocol.get("protocol_sha256")
    if not isinstance(declared, str) or not declared:
        raise RealizabilityProtocolError("realizability protocol missing protocol_sha256")
    computed = protocol_sha256(protocol)
    if declared != computed:
        raise RealizabilityProtocolError(
            f"protocol_sha256 mismatch: declared {declared}, computed {computed}"
        )
    return computed


def load_realizability_protocol(
    path: str | Path = "protocols/frozen_realizability_protocol.json",
) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text())
    require_matching_protocol_sha(protocol)
    return protocol
