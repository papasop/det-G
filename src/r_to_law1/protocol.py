"""Frozen protocol helpers for gate thresholds and protocol hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


class ProtocolError(ValueError):
    """Raised when a frozen protocol is missing required gate inputs."""


REQUIRED_DECISION_THRESHOLDS = {
    "nondegeneracy_tol",
    "hessian_discriminant_tol",
    "null_ray_residual_tol",
    "finite_root_residual_tol",
    "tangent_slope_relative_tol",
    "gl2_covariance_relative_tol",
    "gl2_zero_set_tol",
    "tesc_conformal_residual_tol",
    "two_channel_independence_tol",
    "two_channel_null_residual_tol",
    "centering_value_tol",
    "centering_gradient_tol",
    "ray_equivalence_tol",
    "scale_closure_tol",
    "channel_independence_tol",
    "zero_set_forward_violation_tol",
    "zero_set_reverse_violation_tol",
    "minimum_branch_separation",
}


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
    payload = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def require_decision_thresholds(protocol: dict[str, Any]) -> dict[str, float]:
    thresholds = protocol.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        raise ProtocolError("frozen protocol is missing decision_thresholds")

    missing = sorted(REQUIRED_DECISION_THRESHOLDS - set(thresholds))
    if missing:
        raise ProtocolError(
            "frozen protocol is missing decision thresholds: " + ", ".join(missing)
        )
    return thresholds


def threshold(protocol: dict[str, Any], name: str) -> float:
    thresholds = require_decision_thresholds(protocol)
    try:
        return float(thresholds[name])
    except KeyError as exc:
        raise ProtocolError(f"frozen protocol is missing decision threshold: {name}") from exc


def protocol_sha256(protocol: dict[str, Any]) -> str:
    comparable = dict(protocol)
    comparable.pop("protocol_sha256", None)
    return canonical_hash(comparable)


def require_matching_protocol_sha(protocol: dict[str, Any]) -> str:
    declared = protocol.get("protocol_sha256")
    if not isinstance(declared, str) or not declared:
        raise ProtocolError("frozen protocol is missing protocol_sha256")
    computed = protocol_sha256(protocol)
    if declared != computed:
        raise ProtocolError(
            f"protocol_sha256 mismatch: declared {declared}, computed {computed}"
        )
    return computed
