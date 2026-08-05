"""Path-level Principle-R zero-mode witness audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .certificate import ZeroModeCertificate, source_bound_gates
from .path_cost import accumulated_cost, path_is_nonconstant, positive_measure_fraction
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
        protocol["path_nonconstant_tol"]
    )
    zero_local = np.abs(record["local_costs"]) <= float(protocol["local_cost_zero_tol"])
    fraction = positive_measure_fraction(nonzero_velocity & zero_local, grid)
    return {
        "synthetic_nonconstant_zero_cost_path_pass": path_is_nonconstant(
            path,
            grid,
            float(protocol["path_nonconstant_tol"]),
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
}


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and bool(np.isfinite(float(value)))


def validate_path_record(path_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(path_record, dict):
        return {
            "valid": False,
            "errors": ["path record missing or not a JSON object"],
            "record": {},
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

    return {
        "valid": not errors,
        "errors": errors,
        "record": dict(path_record),
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
    path_data_source_matches_argument = False
    if certificate is not None and path_record_source_path is not None:
        path_data_source_matches_argument = (
            Path(path_record_source_path).resolve()
            == _resolve_declared_path(certificate, certificate_base_dir)
        )
    path_data_source_bound = (
        source_gates["path_data_source_bound"] and path_data_source_matches_argument
    )
    local_positive_measure = bool(
        certificate.local_zero_mode_positive_measure if certificate else False
    )
    local_positive_measure = (
        local_positive_measure
        and path_validation["valid"]
        and float(record["positive_measure_fraction"])
        >= float(protocol["minimum_positive_measure_fraction"])
    )
    total_zero = bool(certificate.zero_total_cost if certificate else False)
    total_zero = (
        total_zero
        and path_validation["valid"]
        and abs(float(record["accumulated_cost"]))
        <= float(protocol["total_cost_zero_tol"])
    )
    cost_nonnegative = (
        bool(certificate and certificate.cost_nonnegative)
        and path_validation["valid"]
        and bool(record["cost_nonnegative"])
    )
    finite_cost_evidence = (
        path_validation["valid"] and bool(record["finite_cost_values"])
    )
    positive_control_nonzero = (
        bool(certificate and certificate.same_meter_positive_control)
        and path_validation["valid"]
        and float(record["same_meter_positive_control_cost"])
        >= float(protocol["minimum_positive_control_cost"])
    )
    gates = {
        "R1_state_admissible_domain_source_bound": all(
            value for key, value in source_gates.items() if key != "path_data_source_bound"
        ),
        "R2_protocol_and_nonnegative_cost_predeclared": bool(
            certificate and certificate.protocol_predeclared
        )
        and protocol_hash_matches
        and cost_nonnegative,
        "R3_contraction_family_gives_zero_infimum": bool(
            certificate
            and certificate.contraction_family_certified
            and certificate.zero_infimum_certified
        ),
        "R4_attained_path_finite_and_nonconstant": bool(
            certificate and certificate.path_nonconstant
        )
        and finite_cost_evidence,
        "R5_accumulated_cost_zero": total_zero,
        "R6_local_zero_mode_positive_measure": local_positive_measure,
        "R7_same_meter_positive_control_nonzero": bool(
            positive_control_nonzero
        ),
        "R8_witness_independent_of_target_G_TESC": bool(
            certificate and certificate.witness_not_constructed_from_target_G
        ),
        "R9_path_data_source_bound": path_data_source_bound,
    }
    return {
        "gates": gates,
        "path_level_R_pipeline_supported": all(
            gates[key]
            for key in (
                "R1_state_admissible_domain_source_bound",
                "R2_protocol_and_nonnegative_cost_predeclared",
                "R3_contraction_family_gives_zero_infimum",
                "R4_attained_path_finite_and_nonconstant",
                "R5_accumulated_cost_zero",
                "R9_path_data_source_bound",
            )
        ),
        "certificate_supplied": certificate is not None,
        "path_data_supplied": path_record is not None,
        "path_record_validation": {
            "valid": path_validation["valid"],
            "errors": path_validation["errors"],
        },
        "source_bound_gates": {
            **source_gates,
            "path_data_source_matches_argument": path_data_source_matches_argument,
        },
        "zero_infimum_certified": gates["R3_contraction_family_gives_zero_infimum"],
        "attained_nonconstant_zero_cost_path": gates[
            "R4_attained_path_finite_and_nonconstant"
        ]
        and gates["R5_accumulated_cost_zero"],
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
