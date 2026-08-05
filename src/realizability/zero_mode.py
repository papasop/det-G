"""Path-level Principle-R zero-mode witness audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .certificate import ZeroModeCertificate, source_bound_gates
from .path_cost import (
    accumulated_cost,
    path_is_nonconstant,
    positive_measure_fraction,
    second_order_path_derivative,
)
from .protocol import protocol_sha256


def _default_grid(protocol: dict[str, Any]) -> np.ndarray:
    interval = protocol["parameter_interval"]
    count = int(protocol.get("parameter_grid_points", 101))
    return np.linspace(float(interval[0]), float(interval[1]), count)


def synthetic_self_test(protocol: dict[str, Any]) -> dict[str, bool]:
    grid = _default_grid(protocol)

    def path(parameter: float) -> np.ndarray:
        return np.array([parameter, 0.0])

    def velocity(_: float) -> np.ndarray:
        return np.array([1.0, 0.0])

    def zero_cost(_x: np.ndarray, _v: np.ndarray) -> float:
        return 0.0

    record = accumulated_cost(
        zero_cost,
        path,
        grid,
        protocol=protocol,
        velocity=velocity,
    )
    nonzero_velocity = np.linalg.norm(record["velocities"], axis=1) > float(
        protocol["velocity_nonzero_tol"]
    )
    zero_local = np.abs(record["local_costs"]) <= float(protocol["local_cost_zero_tol"])
    fraction = positive_measure_fraction(nonzero_velocity & zero_local, grid)
    return {
        "synthetic_nonconstant_zero_cost_path_pass": path_is_nonconstant(
            path,
            grid,
            float(protocol["path_displacement_nonconstant_tol"]),
        )
        and abs(record["accumulated_cost"]) <= float(protocol["total_cost_zero_tol"])
        and fraction >= float(protocol["minimum_positive_measure_fraction"])
    }


REQUIRED_PATH_RECORD_FIELDS = {
    "finite_cost_values",
    "cost_nonnegative",
    "accumulated_cost",
    "positive_measure_fraction",
    "same_meter_positive_control_cost",
    "parameter_grid",
    "path_points",
    "velocities",
    "local_costs",
    "same_meter_positive_control_costs",
}


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and bool(np.isfinite(float(value)))


def _as_float_array(value: Any, *, name: str, ndim: int, errors: list[str]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        errors.append(f"path record field {name!r} must be numeric")
        return np.asarray([])
    if array.ndim != ndim:
        errors.append(f"path record field {name!r} must have dimension {ndim}")
    elif not np.all(np.isfinite(array)):
        errors.append(f"path record field {name!r} must contain finite numbers")
    return array


def validate_path_record(path_record: dict[str, Any] | None) -> dict[str, Any]:
    empty_arrays = {
        "parameter_grid": np.asarray([]),
        "path_points": np.asarray([]),
        "velocities": np.asarray([]),
        "local_costs": np.asarray([]),
        "same_meter_positive_control_costs": np.asarray([]),
    }
    if not isinstance(path_record, dict):
        return {
            "valid": False,
            "errors": ["path record missing or not a JSON object"],
            "record": {},
            "arrays": empty_arrays,
        }

    errors = []
    missing = sorted(REQUIRED_PATH_RECORD_FIELDS - set(path_record))
    if missing:
        errors.append("path record missing fields: " + ", ".join(missing))

    for field in ("finite_cost_values", "cost_nonnegative"):
        if field in path_record and type(path_record[field]) is not bool:
            errors.append(f"path record field {field!r} must be a JSON boolean")

    for field in (
        "accumulated_cost",
        "positive_measure_fraction",
        "same_meter_positive_control_cost",
    ):
        if field in path_record and not _is_finite_number(path_record[field]):
            errors.append(f"path record field {field!r} must be a finite number")

    fraction = path_record.get("positive_measure_fraction")
    if _is_finite_number(fraction) and not 0.0 <= float(fraction) <= 1.0:
        errors.append("path record positive_measure_fraction must lie in [0, 1]")

    parameter_grid = _as_float_array(
        path_record.get("parameter_grid"),
        name="parameter_grid",
        ndim=1,
        errors=errors,
    )
    path_points = _as_float_array(
        path_record.get("path_points"),
        name="path_points",
        ndim=2,
        errors=errors,
    )
    velocities = _as_float_array(
        path_record.get("velocities"),
        name="velocities",
        ndim=2,
        errors=errors,
    )
    local_costs = _as_float_array(
        path_record.get("local_costs"),
        name="local_costs",
        ndim=1,
        errors=errors,
    )
    control_costs = _as_float_array(
        path_record.get("same_meter_positive_control_costs"),
        name="same_meter_positive_control_costs",
        ndim=1,
        errors=errors,
    )
    if parameter_grid.ndim == 1:
        if len(parameter_grid) < 3:
            errors.append("path record parameter_grid must contain at least three samples")
        elif not np.all(np.diff(parameter_grid) > 0):
            errors.append("path record parameter_grid must be strictly increasing")
    sample_count = len(parameter_grid) if parameter_grid.ndim == 1 else -1
    for name, array in (
        ("path_points", path_points),
        ("velocities", velocities),
        ("local_costs", local_costs),
    ):
        if sample_count >= 0 and array.ndim > 0 and len(array) != sample_count:
            errors.append(f"path record field {name!r} length must match parameter_grid")
    if (
        path_points.ndim == 2
        and velocities.ndim == 2
        and path_points.shape != velocities.shape
    ):
        errors.append("path_points and velocities must have the same shape")
    if control_costs.ndim == 1 and len(control_costs) < 1:
        errors.append("same_meter_positive_control_costs must contain at least one value")

    return {
        "valid": not errors,
        "errors": errors,
        "record": dict(path_record),
        "arrays": {
            "parameter_grid": parameter_grid,
            "path_points": path_points,
            "velocities": velocities,
            "local_costs": local_costs,
            "same_meter_positive_control_costs": control_costs,
        },
    }


def _resolve_declared_path(certificate: ZeroModeCertificate, base_dir: str | Path) -> Path:
    declared = Path(certificate.path_data_source_path)
    if not declared.is_absolute():
        declared = Path(base_dir) / declared
    return declared.resolve()


def audit_principle_r_witness(
    protocol: dict[str, Any],
    certificate: ZeroModeCertificate | None = None,
    *,
    certificate_base_dir: str | Path = ".",
    path_record: dict[str, Any] | None = None,
    path_record_source_path: str | Path | None = None,
) -> dict[str, Any]:
    source_gates = (
        source_bound_gates(certificate, base_dir=certificate_base_dir)
        if certificate is not None
        else {
            "cost_source_bound": False,
            "path_source_bound": False,
            "admissible_class_source_bound": False,
            "vacuum_sector_source_bound": False,
            "path_data_source_bound": False,
        }
    )
    protocol_hash_matches = (
        certificate is not None and certificate.protocol_sha256 == protocol_sha256(protocol)
    )
    path_validation = validate_path_record(path_record)
    record = path_validation["record"]
    arrays = path_validation["arrays"]
    path_data_source_matches_argument = False
    if certificate is not None and path_record_source_path is not None:
        path_data_source_matches_argument = (
            Path(path_record_source_path).resolve()
            == _resolve_declared_path(certificate, certificate_base_dir)
        )
    path_data_source_bound = (
        source_gates["path_data_source_bound"] and path_data_source_matches_argument
    )
    computed = {
        "finite_cost_values": False,
        "cost_nonnegative": False,
        "accumulated_cost": None,
        "declared_accumulated_cost_matches_raw": False,
        "path_displacement": None,
        "path_nonconstant": False,
        "frozen_parameter_grid": False,
        "velocity_derivative_max_error": None,
        "velocity_derivative_consistent": False,
        "positive_measure_fraction": None,
        "declared_positive_measure_matches_raw": False,
        "same_meter_positive_control_cost": None,
        "declared_control_cost_matches_raw": False,
    }
    if path_validation["valid"]:
        parameter_grid = arrays["parameter_grid"]
        path_points = arrays["path_points"]
        velocities = arrays["velocities"]
        local_costs = arrays["local_costs"]
        control_costs = arrays["same_meter_positive_control_costs"]
        expected_grid = np.linspace(
            float(protocol["parameter_interval"][0]),
            float(protocol["parameter_interval"][1]),
            int(protocol["parameter_grid_points"]),
        )
        computed["frozen_parameter_grid"] = bool(
            parameter_grid.shape == expected_grid.shape
            and np.allclose(
                parameter_grid,
                expected_grid,
                rtol=float(protocol["parameter_grid_relative_tol"]),
                atol=float(protocol["parameter_grid_absolute_tol"]),
            )
        )
        path_derivatives = second_order_path_derivative(
            parameter_grid,
            path_points,
        )
        derivative_errors = np.linalg.norm(velocities - path_derivatives, axis=1)
        derivative_bounds = float(protocol["velocity_derivative_absolute_tol"]) + float(
            protocol["velocity_derivative_relative_tol"]
        ) * np.linalg.norm(path_derivatives, axis=1)
        computed["velocity_derivative_max_error"] = float(np.max(derivative_errors))
        computed["velocity_derivative_consistent"] = bool(
            np.all(derivative_errors <= derivative_bounds)
        )
        velocity_norms = np.linalg.norm(velocities, axis=1)
        zero_local = np.abs(local_costs) <= float(protocol["local_cost_zero_tol"])
        active = (velocity_norms > float(protocol["velocity_nonzero_tol"])) & zero_local
        computed["finite_cost_values"] = bool(
            np.all(np.isfinite(local_costs)) and np.all(np.isfinite(control_costs))
        )
        computed["cost_nonnegative"] = bool(
            np.min(local_costs) >= -float(protocol["local_cost_zero_tol"])
        )
        computed["accumulated_cost"] = float(np.trapezoid(local_costs, parameter_grid))
        computed["declared_accumulated_cost_matches_raw"] = bool(
            abs(float(record["accumulated_cost"]) - float(computed["accumulated_cost"]))
            <= float(protocol["summary_accumulated_cost_absolute_tol"])
        )
        computed["path_displacement"] = float(
            np.max(np.linalg.norm(path_points - path_points[0], axis=1))
        )
        computed["path_nonconstant"] = bool(
            computed["path_displacement"]
            > float(protocol["path_displacement_nonconstant_tol"])
        )
        computed["positive_measure_fraction"] = positive_measure_fraction(
            active,
            parameter_grid,
        )
        computed["declared_positive_measure_matches_raw"] = bool(
            abs(
                float(record["positive_measure_fraction"])
                - float(computed["positive_measure_fraction"])
            )
            <= float(protocol["summary_positive_measure_absolute_tol"])
        )
        computed["same_meter_positive_control_cost"] = float(np.min(control_costs))
        computed["declared_control_cost_matches_raw"] = bool(
            abs(
                float(record["same_meter_positive_control_cost"])
                - float(computed["same_meter_positive_control_cost"])
            )
            <= float(protocol["summary_control_cost_absolute_tol"])
        )

    finite_cost_evidence = (
        path_validation["valid"]
        and bool(record["finite_cost_values"])
        and bool(computed["finite_cost_values"])
        and bool(computed["frozen_parameter_grid"])
        and bool(computed["velocity_derivative_consistent"])
    )
    cost_nonnegative = (
        bool(certificate and certificate.cost_nonnegative)
        and path_validation["valid"]
        and bool(record["cost_nonnegative"])
        and bool(computed["cost_nonnegative"])
    )
    local_positive_measure = bool(
        certificate.local_zero_mode_positive_measure if certificate else False
    )
    local_positive_measure = (
        local_positive_measure
        and path_validation["valid"]
        and float(record["positive_measure_fraction"])
        >= float(protocol["minimum_positive_measure_fraction"])
        and computed["positive_measure_fraction"] is not None
        and float(computed["positive_measure_fraction"])
        >= float(protocol["minimum_positive_measure_fraction"])
        and bool(computed["declared_positive_measure_matches_raw"])
    )
    total_zero = bool(certificate.zero_total_cost if certificate else False)
    total_zero = (
        total_zero
        and path_validation["valid"]
        and abs(float(record["accumulated_cost"]))
        <= float(protocol["total_cost_zero_tol"])
        and computed["accumulated_cost"] is not None
        and abs(float(computed["accumulated_cost"]))
        <= float(protocol["total_cost_zero_tol"])
        and bool(computed["declared_accumulated_cost_matches_raw"])
    )
    path_nonconstant = (
        bool(certificate and certificate.path_nonconstant)
        and path_validation["valid"]
        and bool(computed["path_nonconstant"])
    )
    positive_control_nonzero = (
        bool(certificate and certificate.same_meter_positive_control)
        and path_validation["valid"]
        and float(record["same_meter_positive_control_cost"])
        >= float(protocol["minimum_positive_control_cost"])
        and computed["same_meter_positive_control_cost"] is not None
        and float(computed["same_meter_positive_control_cost"])
        >= float(protocol["minimum_positive_control_cost"])
        and bool(computed["declared_control_cost_matches_raw"])
    )
    zero_infimum_derived = bool(cost_nonnegative and total_zero)
    gates = {
        "R1_state_admissible_domain_source_bound": all(
            value
            for key, value in source_gates.items()
            if key != "path_data_source_bound"
        ),
        "R2_protocol_and_nonnegative_cost_predeclared": bool(
            certificate and certificate.protocol_predeclared
        )
        and protocol_hash_matches
        and cost_nonnegative,
        "R3_zero_infimum_derived": zero_infimum_derived,
        "R4_path_kinematics_consistent": finite_cost_evidence,
        "R4_attained_path_finite_and_nonconstant": bool(path_nonconstant)
        and finite_cost_evidence,
        "R5_raw_accumulated_cost_zero": total_zero,
        "R5_accumulated_cost_zero": total_zero,
        "R6_raw_local_zero_mode_positive_measure": local_positive_measure,
        "R6_local_zero_mode_positive_measure": local_positive_measure,
        "R7_raw_same_meter_control_positive": bool(positive_control_nonzero),
        "R7_same_meter_positive_control_nonzero": bool(positive_control_nonzero),
        "R8_witness_independent_of_target_G_TESC": bool(
            certificate and certificate.witness_not_constructed_from_target_G
        ),
        "R9_path_data_source_bound": path_data_source_bound,
    }
    return {
        "gates": gates,
        "path_level_R_pipeline_supported": all(gates.values()),
        "certificate_supplied": certificate is not None,
        "path_data_supplied": path_record is not None,
        "path_record_validation": {
            "valid": path_validation["valid"],
            "errors": path_validation["errors"],
        },
        "computed_path_evidence": computed,
        "source_bound_gates": {
            **source_gates,
            "path_data_source_matches_argument": path_data_source_matches_argument,
        },
        "attainment_pipeline_supported": all(
            gates[key]
            for key in (
                "R1_state_admissible_domain_source_bound",
                "R2_protocol_and_nonnegative_cost_predeclared",
                "R3_zero_infimum_derived",
                "R4_path_kinematics_consistent",
                "R4_attained_path_finite_and_nonconstant",
                "R5_raw_accumulated_cost_zero",
                "R9_path_data_source_bound",
            )
        ),
        "path_level_zero_mode_pipeline_supported": all(gates.values()),
        "zero_infimum_certified": gates["R3_zero_infimum_derived"],
        "zero_infimum_derived_from_nonnegative_attained_zero": gates[
            "R3_zero_infimum_derived"
        ],
        "contraction_family_certificate_evidence": bool(
            certificate
            and certificate.contraction_family_certified
            and certificate.zero_infimum_certified
        ),
        "attained_nonconstant_zero_cost_path": gates[
            "R4_attained_path_finite_and_nonconstant"
        ]
        and gates["R5_raw_accumulated_cost_zero"],
        "local_zero_mode_positive_measure": gates[
            "R6_local_zero_mode_positive_measure"
        ],
        "principle_R_witness_source_bound": gates[
            "R1_state_admissible_domain_source_bound"
        ],
        "path_data_source_bound": gates["R9_path_data_source_bound"],
        "principle_R_witness_certified": all(gates.values()),
        "self_tests": synthetic_self_test(protocol),
        "circular_negative_control": bool(
            certificate and not certificate.witness_not_constructed_from_target_G
        ),
    }
