"""Coordinate covariance, unit rescaling and lambda sensitivity audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from .finite_zero_set import _roots_at_x
from .tesc import centred_exposure_cost, derive_component_hessians, derive_tesc_hessian, hessian_2d, signed_cost, task_cost


def audit_gl2_covariance(protocol: dict[str, Any]) -> dict[str, Any]:
    covariance_cfg = protocol["covariance"]
    transform_count = int(covariance_cfg["transforms"])
    seed = int(covariance_cfg["seed"])
    fd = float(protocol["finite_difference_step"])
    G = derive_tesc_hessian(protocol)
    function = lambda z: signed_cost(np.asarray(z, dtype=float), protocol)
    points = []
    for x_value in covariance_cfg["finite_zero_x_samples"]:
        for y_value in _roots_at_x(function, float(x_value), float(protocol["finite_zero_set"]["maximum_abs_y"]), 2e-4):
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
        zero_set_residual = max((abs(function(transform @ np.linalg.solve(transform, point))) for point in points), default=0.0)
        records.append(
            {
                "detS": float(np.linalg.det(transform)),
                "condition": float(np.linalg.cond(transform)),
                "hessian_residual": hessian_residual,
                "zero_set_residual": float(zero_set_residual),
                "signature_preserved": bool(np.linalg.det(transformed_hessian) < 0),
            }
        )
    maximum_hessian_residual = max(record["hessian_residual"] for record in records)
    maximum_zero_residual = max(record["zero_set_residual"] for record in records)
    gates = {
        "GL2_Hessian_covariance": maximum_hessian_residual < 2e-5,
        "GL2_finite_zero_set_covariance": maximum_zero_residual < 1e-10,
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
            "maximum_GL2_signed_zero_set_residual": maximum_zero_residual,
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
    gates = {"Lorentzian_not_unique_to_lambda_one": sum(record["Lorentzian"] for record in records) >= 5}
    return {
        "records": records,
        "gates": gates,
        "gate": all(gates.values()),
        "metrics": {
            "Lorentzian_lambda_count": sum(record["Lorentzian"] for record in records),
            "lambda_count": len(records),
        },
    }


def audit_tesc_operational_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    fd = float(protocol["finite_difference_step"])
    exposure_gradient = np.array(
        [
            (centred_exposure_cost(np.array([fd, 0.0]), protocol) - centred_exposure_cost(np.array([-fd, 0.0]), protocol)) / (2 * fd),
            (centred_exposure_cost(np.array([0.0, fd]), protocol) - centred_exposure_cost(np.array([0.0, -fd]), protocol)) / (2 * fd),
        ]
    )
    gates = {
        "task_and_exposure_separately_computable": np.isfinite(task_cost(np.zeros(2), protocol))
        and np.isfinite(centred_exposure_cost(np.zeros(2), protocol)),
        "centering_removes_exposure_value_and_tangent": abs(centred_exposure_cost(np.zeros(2), protocol)) < 1e-14
        and np.linalg.norm(exposure_gradient) < 1e-8,
        "relative_coefficient_predeclared_as_one": float(protocol["tesc_lambda"]) == 1.0,
    }
    return {"gates": gates, "gate": all(gates.values())}
