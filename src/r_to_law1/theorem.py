"""Two-dimensional conditional theorem and null-ray construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .protocol import threshold


def construct_null_rays(G: np.ndarray, tolerance: float) -> dict[str, Any]:
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


def audit_conditional_theorem(
    G: np.ndarray,
    protocol: dict[str, Any],
    physical_binding_gate: bool | None = None,
) -> dict[str, Any]:
    assumptions = protocol["structural_assumptions"]
    G = np.asarray(G, dtype=float)
    nondegeneracy_tol = threshold(protocol, "nondegeneracy_tol")
    discriminant_tol = threshold(protocol, "hessian_discriminant_tol")
    null_ray_tol = threshold(protocol, "null_ray_residual_tol")
    eigenvalues = np.linalg.eigvalsh((G + G.T) / 2)
    determinant = float(np.linalg.det(G))
    rays = construct_null_rays(G, tolerance=discriminant_tol)
    declared_structural_premises = {
        "principle_R_local_zero_mode_adopted": bool(protocol["principle_R"]["nonzero_zero_cost_direction_required"])
        and bool(assumptions["principle_R_nonzero_direction_attained"]),
        "process_space_is_real_two_dimensional": int(assumptions["selected_process_space_dimension"]) == 2,
        "signed_representative_is_real_C2": bool(assumptions["signed_representative_is_real_C2"]),
        "stationary_basepoint_for_signed_representative": bool(assumptions["stationary_basepoint_for_signed_representative"]),
        "quadratic_form_symmetric": bool(assumptions["signed_representative_is_symmetric"]),
        "quadratic_form_nondegenerate": bool(assumptions["signed_representative_is_nondegenerate"])
        and abs(determinant) > nondegeneracy_tol,
    }
    physical_zero_set_binding = {
        "nonnegative_realization_cost_predeclared": bool(assumptions["nonnegative_realization_cost_predeclared"]),
        "physical_zero_set_equals_signed_representative_zero_set": bool(
            assumptions["physical_zero_set_equals_signed_representative_zero_set"]
        ),
    }
    conclusions = {
        "definite_signed_representative_excluded_if_nonzero_null_vector_exists": True,
        "degenerate_signed_representative_excluded_by_assumption": True,
        "detG_must_be_negative": determinant < 0,
        "signature_must_be_1_1": bool(eigenvalues[0] < 0 < eigenvalues[1]),
        "null_set_is_two_distinct_real_rays": len(rays["null_rays"]) == 2
        and rays["quadratic_discriminant"] > 0
        and max(rays["null_ray_residuals"], default=1.0) < null_ray_tol,
    }
    declared_gate = all(declared_structural_premises.values())
    protocol_binding_gate = all(physical_zero_set_binding.values())
    binding_gate = protocol_binding_gate if physical_binding_gate is None else bool(physical_binding_gate)
    analytic_gate = declared_gate and all(conclusions.values())
    return {
        "declared_structural_premises": declared_structural_premises,
        "physical_zero_set_binding": physical_zero_set_binding,
        "conclusions": conclusions,
        "analytic_theorem_logic_gate": analytic_gate,
        "declared_structural_premises_gate": declared_gate,
        "protocol_zero_set_binding_declaration_gate": protocol_binding_gate,
        "physical_zero_set_binding_certificate_gate": binding_gate,
        "conditional_theorem_premises_gate": analytic_gate and binding_gate,
        "metrics": {
            "G": G,
            "eigenvalues": eigenvalues,
            "detG": determinant,
            **rays,
        },
    }
