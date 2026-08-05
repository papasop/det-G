"""Frozen protocol helpers for path-level realizability audits."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class RealizabilityProtocolError(ValueError):
    """Raised when a realizability protocol is incomplete or modified."""


REQUIRED_PROTOCOL_FIELDS = {
    "schema",
    "version",
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
SUPPORTED_SCHEMA = "realizability-zero-mode-protocol-v0.1.0"
SUPPORTED_VERSION = "0.1.0"
SUPPORTED_DERIVATIVE_METHOD = (
    "second_order_finite_difference_from_path_points_with_velocity_crosscheck"
)
SUPPORTED_INTEGRATION_METHOD = "trapezoid"
NONNEGATIVE_NUMERIC_FIELDS = {
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
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def protocol_sha256(protocol: dict[str, Any]) -> str:
    comparable = dict(protocol)
    comparable.pop("protocol_sha256", None)
    return canonical_hash(comparable)


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_realizability_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != SUPPORTED_SCHEMA:
        raise RealizabilityProtocolError("unsupported realizability protocol schema")
    if protocol.get("version") != SUPPORTED_VERSION:
        raise RealizabilityProtocolError("unsupported realizability protocol version")
    if protocol.get("derivative_method") != SUPPORTED_DERIVATIVE_METHOD:
        raise RealizabilityProtocolError("unsupported realizability derivative_method")
    if protocol.get("integration_method") != SUPPORTED_INTEGRATION_METHOD:
        raise RealizabilityProtocolError("unsupported realizability integration_method")

    interval = protocol.get("parameter_interval")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(_is_finite_number(value) for value in interval)
        or not float(interval[0]) < float(interval[1])
    ):
        raise RealizabilityProtocolError(
            "parameter_interval must contain two finite increasing numbers"
        )

    grid_points = protocol.get("parameter_grid_points")
    if type(grid_points) is not int or grid_points < 3:
        raise RealizabilityProtocolError(
            "parameter_grid_points must be an integer >= 3"
        )

    for field in NONNEGATIVE_NUMERIC_FIELDS:
        value = protocol.get(field)
        if not _is_finite_number(value) or float(value) < 0:
            raise RealizabilityProtocolError(f"{field} must be finite and nonnegative")

    if (
        float(protocol["path_nonconstant_tol"])
        != float(protocol["path_displacement_nonconstant_tol"])
    ):
        raise RealizabilityProtocolError(
            "path_nonconstant_tol is a compatibility alias and must equal "
            "path_displacement_nonconstant_tol"
        )

    fraction = protocol.get("minimum_positive_measure_fraction")
    if not _is_finite_number(fraction) or not 0.0 <= float(fraction) <= 1.0:
        raise RealizabilityProtocolError(
            "minimum_positive_measure_fraction must lie in [0, 1]"
        )

    control = protocol.get("minimum_positive_control_cost")
    if not _is_finite_number(control) or not float(control) > 0:
        raise RealizabilityProtocolError("minimum_positive_control_cost must be > 0")

    declared = protocol.get("protocol_sha256")
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise RealizabilityProtocolError(
            "realizability protocol protocol_sha256 must be 64 lowercase hexadecimal characters"
        )


def require_matching_protocol_sha(protocol: dict[str, Any]) -> str:
    missing = sorted(REQUIRED_PROTOCOL_FIELDS - set(protocol))
    if missing:
        raise RealizabilityProtocolError(
            "realizability protocol missing fields: " + ", ".join(missing)
        )
    _validate_realizability_protocol(protocol)
    declared = protocol.get("protocol_sha256")
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
