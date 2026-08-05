"""Protocol-bound local zero-structure classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .protocol import protocol_sha256, threshold

ZeroSetKind = Literal[
    "empty",
    "trivial",
    "single_line",
    "linear_subspace",
    "two_branch_cone",
    "general_conic_set",
]


@dataclass(frozen=True)
class LocalZeroStructure:
    ambient_dimension: int
    homogeneous_degree: int
    zero_set_kind: ZeroSetKind
    branch_directions: tuple[tuple[float, ...], ...]
    linear_rank: int
    positive_scale_closed: bool
    source_protocol_sha256: str
    provenance_status: str


def normalize_unoriented_ray(
    vector: np.ndarray | list[float] | tuple[float, ...],
    protocol: dict,
) -> tuple[float, ...]:
    ray = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(ray))
    tolerance = threshold(protocol, "ray_equivalence_tol")
    if norm <= tolerance:
        raise ValueError("zero vector cannot define a nontrivial branch")
    ray = ray / norm
    pivot = int(np.argmax(np.abs(ray)))
    if ray[pivot] < 0:
        ray = -ray
    return tuple(float(value) for value in ray)


def _unique_unoriented_rays(
    rays: list[np.ndarray],
    protocol: dict,
) -> tuple[tuple[float, ...], ...]:
    tolerance = threshold(protocol, "ray_equivalence_tol")
    normalized: list[tuple[float, ...]] = []
    for ray in rays:
        candidate = normalize_unoriented_ray(ray, protocol)
        candidate_array = np.asarray(candidate)
        if not any(
            min(
                np.linalg.norm(candidate_array - np.asarray(existing)),
                np.linalg.norm(candidate_array + np.asarray(existing)),
            )
            <= tolerance
            for existing in normalized
        ):
            normalized.append(candidate)
    return tuple(sorted(normalized))


def _kernel_ray(covector: np.ndarray, protocol: dict) -> np.ndarray | None:
    covector = np.asarray(covector, dtype=float)
    tolerance = threshold(protocol, "channel_independence_tol")
    if covector.ndim != 1 or len(covector) != 2:
        raise ValueError("only two-dimensional channel covectors are supported")
    if float(np.linalg.norm(covector)) <= tolerance:
        return None
    return np.array([-covector[1], covector[0]], dtype=float)


def verify_positive_scale_closure(
    branch_directions: tuple[tuple[float, ...], ...],
    protocol: dict,
) -> bool:
    tolerance = threshold(protocol, "scale_closure_tol")
    for direction in branch_directions:
        ray = np.asarray(direction, dtype=float)
        if float(np.linalg.norm(ray)) <= tolerance:
            return False
        for scale in (0.25, 1.0, 4.0):
            normalized = normalize_unoriented_ray(scale * ray, protocol)
            if normalized != normalize_unoriented_ray(ray, protocol):
                return False
    return True


def classify_linear_channels(
    channels: list[np.ndarray] | tuple[np.ndarray, ...],
    protocol: dict,
    *,
    homogeneous_degree: int,
    provenance_status: str,
) -> LocalZeroStructure:
    ambient_dimension = 2
    kernel_rays = [
        ray
        for ray in (_kernel_ray(np.asarray(channel, dtype=float), protocol) for channel in channels)
        if ray is not None
    ]
    branch_directions = _unique_unoriented_rays(kernel_rays, protocol)
    rank = int(np.linalg.matrix_rank(np.vstack(channels))) if channels else 0

    if not branch_directions:
        zero_set_kind: ZeroSetKind = "trivial"
    elif len(branch_directions) == 1:
        zero_set_kind = "single_line"
    elif len(branch_directions) == 2:
        first = np.asarray(branch_directions[0])
        second = np.asarray(branch_directions[1])
        separation = min(
            np.linalg.norm(first - second),
            np.linalg.norm(first + second),
        )
        if separation >= threshold(protocol, "minimum_branch_separation"):
            zero_set_kind = "two_branch_cone"
        else:
            zero_set_kind = "single_line"
    else:
        zero_set_kind = "general_conic_set"

    return LocalZeroStructure(
        ambient_dimension=ambient_dimension,
        homogeneous_degree=homogeneous_degree,
        zero_set_kind=zero_set_kind,
        branch_directions=branch_directions,
        linear_rank=rank,
        positive_scale_closed=verify_positive_scale_closure(branch_directions, protocol),
        source_protocol_sha256=protocol_sha256(protocol),
        provenance_status=provenance_status,
    )


def zero_structure_from_single_channel(
    channel: np.ndarray | list[float],
    protocol: dict,
) -> LocalZeroStructure:
    return classify_linear_channels(
        [np.asarray(channel, dtype=float)],
        protocol,
        homogeneous_degree=1,
        provenance_status="single_channel_analytic",
    )


def zero_structure_from_two_channels(
    channel_plus: np.ndarray | list[float],
    channel_minus: np.ndarray | list[float],
    protocol: dict,
) -> LocalZeroStructure:
    return classify_linear_channels(
        [
            np.asarray(channel_plus, dtype=float),
            np.asarray(channel_minus, dtype=float),
        ],
        protocol,
        homogeneous_degree=2,
        provenance_status="two_channel_analytic",
    )
