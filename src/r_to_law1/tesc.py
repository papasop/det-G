"""Frozen TESC cost and local Hessian reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import require_matching_protocol_sha

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1, -1]).astype(complex)
IDENTITY = np.eye(2, dtype=complex)
INITIAL_STATE = np.array([1, 0], complex)


def load_frozen_protocol(path: str | Path = "protocols/frozen_tesc_protocol.json") -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text())
    require_matching_protocol_sha(protocol)
    return protocol


def unitary_segment(omega: float, detuning: float, phase: float, duration: float) -> np.ndarray:
    vector = np.array(
        [omega * math.cos(phase), omega * math.sin(phase), detuning],
        dtype=float,
    )
    radius = float(np.linalg.norm(vector))
    angle = radius * duration / 2
    if radius < 1e-15:
        return IDENTITY
    generator = vector[0] * SX + vector[1] * SY + vector[2] * SZ
    return math.cos(angle) * IDENTITY - 1j * math.sin(angle) * generator / radius


def endpoint_state(z: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    state = IDENTITY.copy()
    for segment in protocol["segments"]:
        omega = float(segment["omega"]) * (1 + float(z[0]))
        detuning = float(segment["detuning"]) + float(z[1])
        phase = float(segment["phase"])
        duration = float(segment["duration"])
        state = unitary_segment(omega, detuning, phase, duration) @ state
    return state @ INITIAL_STATE


def task_cost(z: np.ndarray, protocol: dict[str, Any]) -> float:
    target = endpoint_state(np.zeros(2), protocol)
    overlap = abs(np.vdot(target, endpoint_state(np.asarray(z, dtype=float), protocol))) ** 2
    return float(max(0.0, 1.0 - overlap))


def rabi_exposure(z: np.ndarray, protocol: dict[str, Any]) -> float:
    z = np.asarray(z, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for segment in protocol["segments"]:
        omega = float(segment["omega"]) * (1 + float(z[0]))
        duration = float(segment["duration"])
        numerator += (omega / 2) ** 2 * duration
        denominator += duration
    return float(numerator / denominator)


def exposure_gradient_at_origin(protocol: dict[str, Any]) -> np.ndarray:
    fd = float(protocol["finite_difference_step"])
    return np.array(
        [
            (rabi_exposure(np.array([fd, 0.0]), protocol) - rabi_exposure(np.array([-fd, 0.0]), protocol)) / (2 * fd),
            (rabi_exposure(np.array([0.0, fd]), protocol) - rabi_exposure(np.array([0.0, -fd]), protocol)) / (2 * fd),
        ],
        dtype=float,
    )


def centred_exposure_cost(z: np.ndarray, protocol: dict[str, Any]) -> float:
    z = np.asarray(z, dtype=float)
    origin = rabi_exposure(np.zeros(2), protocol)
    gradient = exposure_gradient_at_origin(protocol)
    return float(rabi_exposure(z, protocol) - origin - gradient @ z)


def signed_cost(z: np.ndarray, protocol: dict[str, Any], lambda_value: float | None = None) -> float:
    if lambda_value is None:
        lambda_value = float(protocol["tesc_lambda"])
    return float(task_cost(np.asarray(z, dtype=float), protocol) - lambda_value * centred_exposure_cost(z, protocol))


def hessian_2d(function, fd: float) -> np.ndarray:
    origin = np.zeros(2)
    value = function(origin)
    hessian = np.zeros((2, 2), dtype=float)
    for index in range(2):
        step = np.zeros(2)
        step[index] = fd
        hessian[index, index] = (function(step) - 2 * value + function(-step)) / fd**2
    x_step = np.array([fd, 0.0])
    y_step = np.array([0.0, fd])
    hessian[0, 1] = hessian[1, 0] = (
        function(x_step + y_step)
        - function(x_step - y_step)
        - function(-x_step + y_step)
        + function(-x_step - y_step)
    ) / (4 * fd * fd)
    return hessian


def derive_tesc_hessian(protocol: dict[str, Any], lambda_value: float | None = None) -> np.ndarray:
    fd = float(protocol["finite_difference_step"])
    return hessian_2d(lambda z: signed_cost(z, protocol, lambda_value), fd)


def derive_component_hessians(protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    fd = float(protocol["finite_difference_step"])
    return {
        "task": hessian_2d(lambda z: task_cost(z, protocol), fd),
        "centred_exposure": hessian_2d(lambda z: centred_exposure_cost(z, protocol), fd),
        "signed": derive_tesc_hessian(protocol),
    }
