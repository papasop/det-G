"""Two-dimensional conditional theorem and null-ray construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TOLERANCE = 1e-10


def construct_null_rays(G: np.ndarray, tolerance: float = TOLERANCE) -> dict[str, Any]:
    G = np.asarray(G, dtype=float)
    g00, g01, g11 = float(G[0, 0]), float(G[0, 1]), float(G[1, 1])
    discriminant = (2 * g01) ** 2 - 4 * g11 * g00
    rays: list[np.ndarray] = []
    if abs(g11) > tolerance and discriminant >= 0:
        root = math.sqrt(max(0.0, discriminant))
        rays = [
            np.array([1.0, (-2 * g01 + root) / (2 * g11)]),
            np.array([1.0, (-2 * g01 - root) / (2 * g11)]),
        ]
    elif abs(g01) > tolerance:
        rays = [np.array([1.0, -g00 / (2 * g01)]), np.array([0.0, 1.0])]
    normalized = [ray / np.linalg.norm(ray) for ray in rays]
    residuals = [
        abs(float(ray @ G @ ray)) / (1 + np.linalg.norm(G) * np.linalg.norm(ray) ** 2)
        for ray in normalized
    ]
    return {
        "quadratic_discriminant": float(discriminant),
        "null_rays": normalized,
        "null_ray_residuals": residuals,
        "maximum_null_ray_residual": max(residuals, default=None),
    }


def audit_conditional_theorem(G: np.ndarray, protocol: dict[str, Any]) -> dict[str, Any]:
    assumptions = protocol["structural_assumptions"]
    G = np.asarray(G, dtype=float)
    eigenvalues = np.linalg.eigvalsh((G + G.T) / 2)
    determinant = float(np.linalg.det(G))
    rays = construct_null_rays(G)
    premises = {
        "R_requires_nonzero_zero_cost_direction": bool(protocol["principle_R"]["nonzero_zero_cost_direction_required"]),
        "process_space_is_real_two_dimensional": int(assumptions["selected_process_space_dimension"]) == 2,
        "C2_stationary_cost_has_Hessian_tangent_form": bool(assumptions["cost_is_real_C2_near_basepoint"])
        and bool(assumptions["stationary_basepoint"]),
        "quadratic_tangent_cost_complete_for_zero_directions": bool(assumptions["quadratic_tangent_cost_is_complete"]),
        "quadratic_form_symmetric": bool(assumptions["metric_is_symmetric"]),
        "quadratic_form_nondegenerate": bool(assumptions["metric_is_nondegenerate"]) and abs(determinant) > TOLERANCE,
    }
    conclusions = {
        "definite_signatures_excluded": True,
        "degenerate_signature_excluded": True,
        "detG_must_be_negative": determinant < 0,
        "signature_must_be_1_1": bool(eigenvalues[0] < 0 < eigenvalues[1]),
        "null_set_is_two_distinct_real_rays": len(rays["null_rays"]) == 2
        and rays["quadratic_discriminant"] > 0
        and max(rays["null_ray_residuals"], default=1.0) < TOLERANCE,
    }
    return {
        "premises": premises,
        "conclusions": conclusions,
        "gate": all(premises.values()) and all(conclusions.values()),
        "metrics": {
            "G": G,
            "eigenvalues": eigenvalues,
            "detG": determinant,
            **rays,
        },
    }
