"""Coordinate covariance, unit rescaling and lambda sensitivity audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from .finite_zero_set import _roots_at_x
from .protocol import threshold
from .tesc import centred_exposure_cost, derive_component_hessians, derive_tesc_hessian, hessian_2d, signed_cost, task_cost


def audit_gl2_covariance(protocol: dict[str, Any]) -> dict[str, Any]:
    covariance_cfg = protocol["covariance"]
    transform_count = int(covariance_cfg["transforms"])
    seed = int(covariance_cfg["seed"])
    fd = float(protocol["finite_difference_step"])
    hessian_tol = threshold(protocol, "gl2_covariance_relative_tol")
    zero_set_tol = threshold(protocol, "gl2_zero_set_tol")
    G = derive_tesc_hessian(protocol)
    function = lambda z: signed_cost(np.asarray(z, dtype=float), protocol)
    points = []
    for x_value in covariance_cfg["finite_zero_x_samples"]:
        for y_value in _roots_at_x(
            function,
            float(x_value),
            float(protocol["finite_zero_set"]["maximum_abs_y"]),
            float(protocol["finite_zero_set"]["scan_step"]),
        ):
            points.append(np.array([float(x_value), y_value]))
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(transform_count):
        while True:
            transform = rng.normal(size=(2, 2))
            if abs(np.linalg.det(transform)) > 0.25 and np.linalg.cond(transform) < 12:
                break
        transformed_fd = fd / max(np.linalg.norm(transform, 2), 1.0)
        transformed_hessian = hessian_2d(lambda y: function(transform @ np.asarray(y)), transformed_fd)
        target = transform.T @ G @ transform
        hessian_residual = float(np.linalg.norm(transformed_hessian - target) / max(np.linalg.norm(target), 1e-15))
        pullback_identity_residual = max(
            (abs(function(transform @ np.linalg.solve(transform, point))) for point in points),
            default=0.0,
        )
        records.append(
            {
                "detS": float(np.linalg.det(transform)),
                "condition": float(np.linalg.cond(transform)),
                "hessian_residual": hessian_residual,
                "zero_set_pullback_identity_residual": float(pullback_identity_residual),
                "signature_preserved": bool(np.linalg.det(transformed_hessian) < 0),
            }
        )
    maximum_hessian_residual = max(record["hessian_residual"] for record in records)
    maximum_pullback_identity_residual = max(
        record["zero_set_pullback_identity_residual"] for record in records
    )
    gates = {
        "GL2_Hessian_covariance": maximum_hessian_residual < hessian_tol,
        "GL2_zero_set_pullback_identity": maximum_pullback_identity_residual < zero_set_tol,
        "signature_preserved_all_GL2_trials": all(record["signature_preserved"] for record in records),
    }
    return {
        "records": records,
        "gates": gates,
        "gate": all(gates.values()),
        "metrics": {
            "signed_zero_contrast_points": len(points),
            "GL2_trials": len(records),
            "maximum_GL2_Hessian_relative_residual": maximum_hessian_residual,
            "maximum_GL2_zero_set_pullback_identity_residual": maximum_pullback_identity_residual,
        },
    }


def audit_unit_rescaling(protocol: dict[str, Any]) -> dict[str, Any]:
    G = derive_tesc_hessian(protocol)
    records = []
    for scale_omega, scale_detuning in protocol["covariance"]["unit_rescalings"]:
        transform = np.diag([scale_omega, scale_detuning])
        transformed = transform.T @ G @ transform
        records.append(
            {
                "scales": [scale_omega, scale_detuning],
                "det_transformed_G": float(np.linalg.det(transformed)),
                "signature_preserved": bool(np.linalg.det(transformed) < 0),
                "analytic_covariance_residual": float(np.linalg.norm(transformed - transform.T @ G @ transform)),
            }
        )
    gates = {"signature_preserved_under_extreme_unit_rescaling": all(record["signature_preserved"] for record in records)}
    return {"records": records, "gates": gates, "gate": all(gates.values())}


def audit_lambda_sensitivity(protocol: dict[str, Any]) -> dict[str, Any]:
    component_hessians = derive_component_hessians(protocol)
    task_hessian = component_hessians["task"]
    exposure_hessian = component_hessians["centred_exposure"]
    lambda_cfg = protocol["covariance"]["lambda_stability"]
    lambda_one = float(protocol["tesc_lambda"])
    records = []
    for lambda_value in protocol["covariance"]["lambda_grid"]:
        G_lambda = task_hessian - float(lambda_value) * exposure_hessian
        records.append(
            {
                "lambda": lambda_value,
                "detG": float(np.linalg.det(G_lambda)),
                "eigenvalues": np.linalg.eigvalsh(G_lambda),
                "Lorentzian": bool(np.linalg.det(G_lambda) < 0),
            }
        )
    determinant_polynomial = np.poly1d(
        np.polyfit(
            [0.0, 1.0, 2.0],
            [
                float(np.linalg.det(task_hessian - value * exposure_hessian))
                for value in (0.0, 1.0, 2.0)
            ],
            2,
        )
    )
    roots = [
        float(root.real)
        for root in np.roots(determinant_polynomial)
        if abs(float(root.imag)) < 1e-8
    ]
    nearest_boundary_distance = min(
        (abs(root - lambda_one) for root in roots),
        default=float("inf"),
    )
    lambda_one_lorentzian = float(determinant_polynomial(lambda_one)) < 0
    analytic_open_interval = lambda_one_lorentzian and nearest_boundary_distance > 0
    lorentzian_count = sum(record["Lorentzian"] for record in records)
    tested_below = any(
        record["lambda"] < lambda_one and record["Lorentzian"] for record in records
    )
    tested_above = any(
        record["lambda"] > lambda_one and record["Lorentzian"] for record in records
    )
    gates = {
        "lambda_one_Lorentzian": lambda_one_lorentzian,
        "analytic_open_Lorentzian_interval_around_one": analytic_open_interval,
        "Lorentzian_not_unique_to_lambda_one": analytic_open_interval,
        "tested_grid_has_required_below_lambda_one": (
            tested_below
            if bool(lambda_cfg["require_lorentzian_below_tested"])
            else True
        ),
        "tested_grid_has_required_above_lambda_one": (
            tested_above
            if bool(lambda_cfg["require_lorentzian_above_tested"])
            else True
        ),
        "tested_grid_has_minimum_distinct_Lorentzian_lambda_count": lorentzian_count
        >= int(lambda_cfg["minimum_distinct_lorentzian_lambda_count"]),
    }
    return {
        "records": records,
        "gates": gates,
        "gate": all(gates.values()),
        "metrics": {
            "Lorentzian_lambda_count": lorentzian_count,
            "lambda_count": len(records),
            "lambda_det_zero_boundaries": sorted(roots),
            "lambda_one_nearest_det_zero_boundary_distance": nearest_boundary_distance,
        },
    }


def audit_tesc_operational_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    fd = float(protocol["finite_difference_step"])
    centering_value_tol = threshold(protocol, "centering_value_tol")
    centering_gradient_tol = threshold(protocol, "centering_gradient_tol")
    exposure_gradient = np.array(
        [
            (centred_exposure_cost(np.array([fd, 0.0]), protocol) - centred_exposure_cost(np.array([-fd, 0.0]), protocol)) / (2 * fd),
            (centred_exposure_cost(np.array([0.0, fd]), protocol) - centred_exposure_cost(np.array([0.0, -fd]), protocol)) / (2 * fd),
        ]
    )
    gates = {
        "task_and_exposure_separately_computable": np.isfinite(task_cost(np.zeros(2), protocol))
        and np.isfinite(centred_exposure_cost(np.zeros(2), protocol)),
        "centering_removes_exposure_value_and_tangent": abs(centred_exposure_cost(np.zeros(2), protocol)) < centering_value_tol
        and np.linalg.norm(exposure_gradient) < centering_gradient_tol,
        "relative_coefficient_predeclared_as_one": float(protocol["tesc_lambda"]) == 1.0,
    }
    return {"gates": gates, "gate": all(gates.values())}
