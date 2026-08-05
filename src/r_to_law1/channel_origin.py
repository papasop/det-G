"""Analytic single-channel and two-channel zero-cone interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import threshold
from .zero_structure import (
    zero_structure_from_single_channel,
    zero_structure_from_two_channels,
)


def induced_quadratic_form(
    channel_plus: np.ndarray | list[float],
    channel_minus: np.ndarray | list[float],
) -> np.ndarray:
    plus = np.asarray(channel_plus, dtype=float)
    minus = np.asarray(channel_minus, dtype=float)
    return 0.5 * (np.outer(plus, minus) + np.outer(minus, plus))


def analytic_two_channel_determinant_identity(
    channel_plus: np.ndarray | list[float],
    channel_minus: np.ndarray | list[float],
    protocol: dict,
) -> dict[str, Any]:
    plus = np.asarray(channel_plus, dtype=float)
    minus = np.asarray(channel_minus, dtype=float)
    channel_matrix = np.vstack([plus, minus])
    channel_determinant = float(np.linalg.det(channel_matrix))
    G = induced_quadratic_form(plus, minus)
    det_G = float(np.linalg.det(G))
    expected = -0.25 * channel_determinant**2
    residual = abs(det_G - expected)
    tolerance = threshold(protocol, "channel_independence_tol")
    eigenvalues = np.linalg.eigvalsh(G)
    return {
        "G": G,
        "channel_matrix": channel_matrix,
        "channel_determinant": channel_determinant,
        "detG": det_G,
        "expected_detG": expected,
        "determinant_identity_residual": residual,
        "determinant_identity_verified": residual <= tolerance,
        "channels_independent": abs(channel_determinant) > tolerance,
        "signature_1_1": bool(eigenvalues[0] < 0 < eigenvalues[1]),
        "eigenvalues": eigenvalues,
    }


def audit_single_channel_no_go(
    channel: np.ndarray | list[float],
    protocol: dict,
) -> dict[str, Any]:
    structure = zero_structure_from_single_channel(channel, protocol)
    return {
        "F_single": "abs(L(v))/H with H>0",
        "zero_structure": structure,
        "single_channel_no_go_certified": structure.zero_set_kind == "single_line",
        "two_branch_zero_cone_supported": False,
        "interpretation": (
            "A nonzero scalar linear channel on a two-dimensional plane has one "
            "kernel line, not two distinct Lorentzian null rays."
        ),
    }


def audit_two_channel_origin(
    channel_plus: np.ndarray | list[float],
    channel_minus: np.ndarray | list[float],
    protocol: dict,
) -> dict[str, Any]:
    structure = zero_structure_from_two_channels(channel_plus, channel_minus, protocol)
    identity = analytic_two_channel_determinant_identity(
        channel_plus,
        channel_minus,
        protocol,
    )
    gate = (
        structure.zero_set_kind == "two_branch_cone"
        and identity["channels_independent"]
        and identity["determinant_identity_verified"]
        and identity["detG"] < 0
        and identity["signature_1_1"]
    )
    return {
        "F_double": "abs(L_plus(v) L_minus(v))/H with H>0",
        "zero_structure": structure,
        "quadratic_identity": identity,
        "Z_F_double": "ker L_plus union ker L_minus",
        "two_channel_complexity_required": True,
        "two_branch_zero_cone_supported": gate,
        "gate": gate,
    }
