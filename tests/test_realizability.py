from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from realizability.certificate import (
    ZeroModeCertificate,
    ZeroModeCertificateError,
    load_zero_mode_certificate,
)
from realizability.path_cost import (
    accumulated_cost,
    path_is_nonconstant,
    positive_measure_fraction,
    second_order_path_derivative,
)
from realizability.protocol import (
    RealizabilityProtocolError,
    load_realizability_protocol,
    protocol_sha256,
)
from realizability.zero_mode import audit_principle_r_witness
from r_to_law1.tesc import derive_tesc_hessian, load_frozen_protocol
from r_to_law1.theorem import audit_conditional_theorem


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RealizabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_realizability_protocol()
        self.grid = np.linspace(0.0, 1.0, 11)

    def _write_path_data(self, directory: Path, **overrides) -> Path:
        parameter_grid = np.linspace(
            self.protocol["parameter_interval"][0],
            self.protocol["parameter_interval"][1],
            self.protocol["parameter_grid_points"],
        )
        path_points = [[float(t), 0.0] for t in parameter_grid]
        velocities = [[1.0, 0.0] for _ in parameter_grid]
        zero_costs = [0.0 for _ in parameter_grid]
        control_costs = [1.0 for _ in parameter_grid]
        data = {
            "finite_cost_values": True,
            "cost_nonnegative": True,
            "accumulated_cost": 0.0,
            "positive_measure_fraction": 1.0,
            "same_meter_positive_control_cost": 1.0,
            "parameter_grid": parameter_grid.tolist(),
            "path_points": path_points,
            "velocities": velocities,
            "local_costs": zero_costs,
            "same_meter_positive_control_costs": control_costs,
        }
        data.update(overrides)
        path = directory / "path_data.json"
        path.write_text(json.dumps(data, sort_keys=True) + "\n")
        return path

    def _write_polynomial_path_data(self, directory: Path, power: int) -> Path:
        parameter_grid = np.linspace(
            self.protocol["parameter_interval"][0],
            self.protocol["parameter_interval"][1],
            self.protocol["parameter_grid_points"],
        )
        path_points = [[float(t**power), 0.0] for t in parameter_grid]
        velocities = [[float(power * t ** (power - 1)), 0.0] for t in parameter_grid]
        active = parameter_grid > 0.0
        positive_measure_fraction = float(
            np.sum(np.diff(parameter_grid)[active[:-1] & active[1:]])
            / (parameter_grid[-1] - parameter_grid[0])
        )
        return self._write_path_data(
            directory,
            positive_measure_fraction=positive_measure_fraction,
            path_points=path_points,
            velocities=velocities,
        )

    def _source_bound_certificate_data(
        self,
        directory: Path,
        path_data: Path,
        **overrides,
    ) -> dict:
        files = {}
        for name in ("cost", "path", "admissible", "vacuum"):
            source = directory / f"{name}.txt"
            source.write_text(f"{name} source\n")
            files[name] = (source.name, sha256(source))
        data = {
            "schema_version": "zero-mode-certificate-v0.1.0",
            "protocol_sha256": self.protocol["protocol_sha256"],
            "cost_source_path": files["cost"][0],
            "cost_source_sha256": files["cost"][1],
            "path_source_path": files["path"][0],
            "path_source_sha256": files["path"][1],
            "admissible_class_source_path": files["admissible"][0],
            "admissible_class_source_sha256": files["admissible"][1],
            "vacuum_sector_source_path": files["vacuum"][0],
            "vacuum_sector_source_sha256": files["vacuum"][1],
            "path_data_source_path": path_data.name,
            "path_data_source_sha256": sha256(path_data),
            "protocol_predeclared": True,
            "cost_nonnegative": True,
            "contraction_family_certified": True,
            "zero_infimum_certified": True,
            "path_nonconstant": True,
            "zero_total_cost": True,
            "local_zero_mode_positive_measure": True,
            "same_meter_positive_control": True,
            "witness_not_constructed_from_target_G": True,
        }
        data.update(overrides)
        return data

    def _source_bound_certificate(
        self,
        directory: Path,
        path_data: Path,
        **overrides,
    ) -> ZeroModeCertificate:
        return ZeroModeCertificate.from_dict(
            self._source_bound_certificate_data(directory, path_data, **overrides)
        )

    def _audit_with(self, directory: Path, cert: ZeroModeCertificate, path_data: Path) -> dict:
        return audit_principle_r_witness(
            self.protocol,
            cert,
            certificate_base_dir=directory,
            path_record=json.loads(path_data.read_text()),
            path_record_source_path=path_data,
        )

    def _write_rehashed_protocol(self, directory: Path, **overrides) -> Path:
        protocol = dict(self.protocol)
        protocol.update(overrides)
        protocol["protocol_sha256"] = protocol_sha256(protocol)
        path = directory / "protocol.json"
        path.write_text(json.dumps(protocol, sort_keys=True) + "\n")
        return path

    def test_valid_upstream_zero_mode_certificate_gate_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertTrue(result["principle_R_witness_certified"])
        self.assertTrue(result["path_data_source_bound"])

    def test_quadratic_path_with_analytic_velocity_passes_endpoint_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_polynomial_path_data(directory, power=2)
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertTrue(
            result["computed_path_evidence"]["velocity_derivative_consistent"]
        )
        self.assertTrue(result["gates"]["R4_path_kinematics_consistent"])
        self.assertTrue(result["principle_R_witness_certified"])

    def test_cubic_path_derivative_error_has_second_order_scaling(self) -> None:
        coarse_grid = np.linspace(0.0, 1.0, 101)
        fine_grid = np.linspace(0.0, 1.0, 201)
        coarse_points = np.column_stack((coarse_grid**3, np.zeros_like(coarse_grid)))
        fine_points = np.column_stack((fine_grid**3, np.zeros_like(fine_grid)))
        coarse_velocity = np.column_stack((3 * coarse_grid**2, np.zeros_like(coarse_grid)))
        fine_velocity = np.column_stack((3 * fine_grid**2, np.zeros_like(fine_grid)))
        coarse_error = np.max(
            np.linalg.norm(
                second_order_path_derivative(coarse_grid, coarse_points)
                - coarse_velocity,
                axis=1,
            )
        )
        fine_error = np.max(
            np.linalg.norm(
                second_order_path_derivative(fine_grid, fine_points)
                - fine_velocity,
                axis=1,
            )
        )
        self.assertLess(coarse_error, self.protocol["velocity_derivative_absolute_tol"])
        self.assertLess(fine_error, coarse_error / 3.5)

    def test_nonlinear_endpoint_does_not_cause_correct_path_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_polynomial_path_data(directory, power=3)
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertTrue(
            result["computed_path_evidence"]["velocity_derivative_consistent"]
        )
        self.assertTrue(result["principle_R_witness_certified"])

    def test_accumulated_cost_uses_second_order_fallback_for_nonlinear_path(self) -> None:
        grid = np.linspace(0.0, 1.0, self.protocol["parameter_grid_points"])

        def path(parameter: float) -> np.ndarray:
            return np.array([parameter**2, 0.0])

        def cost(_x: np.ndarray, velocity: np.ndarray) -> float:
            return float(np.linalg.norm(velocity))

        record = accumulated_cost(cost, path, grid, protocol=self.protocol)
        self.assertLess(np.linalg.norm(record["velocities"][0]), 1e-12)
        self.assertAlmostEqual(record["velocities"][-1][0], 2.0)
        self.assertAlmostEqual(record["accumulated_cost"], 1.0, places=12)

    def test_accumulated_cost_rejects_two_point_second_order_grid(self) -> None:
        grid = np.array([0.0, 1.0])
        path = lambda parameter: np.array([parameter, 0.0])
        cost = lambda _x, _v: 0.0
        with self.assertRaises(ValueError):
            accumulated_cost(cost, path, grid, protocol=self.protocol)

    def test_zero_infimum_is_derived_not_self_certified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(
                directory,
                path_data,
                contraction_family_certified=False,
                zero_infimum_certified=False,
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertTrue(result["gates"]["R3_zero_infimum_derived"])
        self.assertFalse(result["contraction_family_certificate_evidence"])
        self.assertTrue(result["principle_R_witness_certified"])

    def test_rehashed_protocol_with_two_point_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_rehashed_protocol(
                Path(tmpdir),
                parameter_grid_points=2,
            )
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_rehashed_protocol_with_negative_tolerance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_rehashed_protocol(
                Path(tmpdir),
                velocity_derivative_absolute_tol=-1e-6,
            )
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_rehashed_protocol_with_nan_tolerance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_rehashed_protocol(
                Path(tmpdir),
                velocity_derivative_absolute_tol=float("nan"),
            )
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_rehashed_protocol_with_unsupported_methods_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_rehashed_protocol(
                Path(tmpdir),
                derivative_method="caller_velocity_or_central_difference",
            )
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_rehashed_protocol(
                Path(tmpdir),
                integration_method="simpson",
            )
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_rehashed_protocol_with_unsupported_identity_fields_is_rejected(self) -> None:
        cases = [
            {"schema": "realizability-zero-mode-protocol-v9.9.9"},
            {"version": "9.9.9"},
            {"contraction_sequence_acceptance_rule": "accept_any_contraction"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = self._write_rehashed_protocol(Path(tmpdir), **overrides)
                    with self.assertRaises(RealizabilityProtocolError):
                        load_realizability_protocol(path)

    def test_rehashed_protocol_with_invalid_interval_is_rejected(self) -> None:
        cases = [
            {"parameter_interval": [0.0]},
            {"parameter_interval": [1.0, 0.0]},
            {"parameter_interval": [0.0, float("inf")]},
            {"parameter_interval": "0,1"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = self._write_rehashed_protocol(Path(tmpdir), **overrides)
                    with self.assertRaises(RealizabilityProtocolError):
                        load_realizability_protocol(path)

    def test_rehashed_protocol_with_invalid_fraction_or_control_is_rejected(self) -> None:
        cases = [
            {"minimum_positive_measure_fraction": -0.1},
            {"minimum_positive_measure_fraction": 1.1},
            {"minimum_positive_measure_fraction": float("nan")},
            {"minimum_positive_control_cost": 0.0},
            {"minimum_positive_control_cost": -1e-9},
            {"minimum_positive_control_cost": float("inf")},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = self._write_rehashed_protocol(Path(tmpdir), **overrides)
                    with self.assertRaises(RealizabilityProtocolError):
                        load_realizability_protocol(path)

    def test_rehashed_protocol_with_alias_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_rehashed_protocol(
                Path(tmpdir),
                path_nonconstant_tol=2e-10,
            )
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_certificate_supplied_without_path_data_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(directory, path_data)
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=directory,
                path_record=None,
                path_record_source_path=None,
            )
        self.assertFalse(result["principle_R_witness_certified"])
        self.assertFalse(result["path_record_validation"]["valid"])

    def test_path_data_supplied_without_certificate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_data = self._write_path_data(Path(tmpdir))
            result = audit_principle_r_witness(
                self.protocol,
                None,
                path_record=json.loads(path_data.read_text()),
                path_record_source_path=path_data,
            )
        self.assertFalse(result["principle_R_witness_certified"])

    def test_missing_path_record_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            record = json.loads(path_data.read_text())
            record.pop("accumulated_cost")
            path_data.write_text(json.dumps(record) + "\n")
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["path_record_validation"]["valid"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_certificate_string_false_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            data = self._source_bound_certificate_data(
                directory,
                path_data,
                cost_nonnegative="false",
            )
            with self.assertRaises(ZeroModeCertificateError):
                ZeroModeCertificate.from_dict(data)

    def test_path_record_string_false_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory, cost_nonnegative="false")
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["path_record_validation"]["valid"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_unknown_certificate_schema_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            data = self._source_bound_certificate_data(
                directory,
                path_data,
                schema_version="zero-mode-certificate-v9.9.9",
            )
            with self.assertRaises(ZeroModeCertificateError):
                ZeroModeCertificate.from_dict(data)

    def test_malformed_sha_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            data = self._source_bound_certificate_data(
                directory,
                path_data,
                cost_source_sha256="A" * 64,
            )
            with self.assertRaises(ZeroModeCertificateError):
                ZeroModeCertificate.from_dict(data)

    def test_forged_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(
                directory,
                path_data,
                cost_source_sha256="a" * 64,
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["principle_R_witness_source_bound"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_stale_protocol_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(
                directory,
                path_data,
                protocol_sha256="b" * 64,
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["gates"]["R2_protocol_and_nonnegative_cost_predeclared"])

    def test_path_data_change_after_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(directory, path_data)
            path_data.write_text(
                json.dumps(
                    {
                        "finite_cost_values": True,
                        "cost_nonnegative": True,
                        "accumulated_cost": 0.0,
                        "positive_measure_fraction": 1.0,
                        "same_meter_positive_control_cost": 2.0,
                    }
                )
                + "\n"
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["path_data_source_bound"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_non_frozen_parameter_grid_fails_kinematics_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            grid = np.linspace(0.0, 1.0, 100)
            path_data = self._write_path_data(
                directory,
                parameter_grid=grid.tolist(),
                path_points=[[float(t), 0.0] for t in grid],
                velocities=[[1.0, 0.0] for _ in grid],
                local_costs=[0.0 for _ in grid],
                same_meter_positive_control_costs=[1.0 for _ in grid],
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["computed_path_evidence"]["frozen_parameter_grid"])
        self.assertFalse(result["gates"]["R4_path_kinematics_consistent"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_velocity_not_matching_path_derivative_fails_kinematics_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                velocities=[
                    [2.0, 0.0]
                    for _ in range(self.protocol["parameter_grid_points"])
                ],
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(
            result["computed_path_evidence"]["velocity_derivative_consistent"]
        )
        self.assertFalse(result["gates"]["R4_path_kinematics_consistent"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_negative_local_cost_fails_nonnegativity_gate(self) -> None:
        path = lambda t: np.array([t, 0.0])
        velocity = lambda _t: np.array([1.0, 0.0])
        cost = lambda _x, _v: -1e-3
        record = accumulated_cost(
            cost,
            path,
            self.grid,
            protocol=self.protocol,
            velocity=velocity,
        )
        self.assertFalse(record["cost_nonnegative"])
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                cost_nonnegative=record["cost_nonnegative"],
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["gates"]["R2_protocol_and_nonnegative_cost_predeclared"])

    def test_nonzero_accumulated_cost_fails_zero_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory, accumulated_cost=1.0)
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["gates"]["R5_accumulated_cost_zero"])

    def test_zero_summary_with_nonzero_raw_local_cost_fails_zero_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                accumulated_cost=0.0,
                local_costs=[1.0 for _ in range(self.protocol["parameter_grid_points"])],
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertGreater(result["computed_path_evidence"]["accumulated_cost"], 0.0)
        self.assertFalse(result["gates"]["R5_accumulated_cost_zero"])

    def test_accumulated_cost_summary_mismatch_fails_even_when_both_zero_like(self) -> None:
        declared_accumulated_cost = 5e-11
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                accumulated_cost=declared_accumulated_cost,
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertLess(
            abs(declared_accumulated_cost),
            self.protocol["total_cost_zero_tol"],
        )
        self.assertFalse(
            result["computed_path_evidence"]["declared_accumulated_cost_matches_raw"]
        )
        self.assertFalse(result["gates"]["R5_raw_accumulated_cost_zero"])

    def test_constant_path_declaration_fails_closed(self) -> None:
        path = lambda _t: np.array([1.0, 0.0])
        self.assertFalse(path_is_nonconstant(path, self.grid, 1e-10))
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(
                directory,
                path_data,
                path_nonconstant=False,
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["gates"]["R4_attained_path_finite_and_nonconstant"])

    def test_constant_raw_path_with_true_declaration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                path_points=[
                    [1.0, 0.0]
                    for _ in range(self.protocol["parameter_grid_points"])
                ],
                velocities=[
                    [0.0, 0.0]
                    for _ in range(self.protocol["parameter_grid_points"])
                ],
            )
            cert = self._source_bound_certificate(
                directory,
                path_data,
                path_nonconstant=True,
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["computed_path_evidence"]["path_nonconstant"])
        self.assertFalse(result["gates"]["R4_attained_path_finite_and_nonconstant"])
        self.assertFalse(result["principle_R_witness_certified"])

    def test_positive_measure_fraction_below_threshold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory, positive_measure_fraction=0.0)
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["gates"]["R6_local_zero_mode_positive_measure"])

    def test_positive_measure_summary_without_raw_support_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                positive_measure_fraction=1.0,
                velocities=[
                    [0.0, 0.0]
                    for _ in range(self.protocol["parameter_grid_points"])
                ],
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertEqual(result["computed_path_evidence"]["positive_measure_fraction"], 0.0)
        self.assertFalse(result["gates"]["R6_local_zero_mode_positive_measure"])
        self.assertFalse(result["path_level_R_pipeline_supported"])
        self.assertFalse(result["path_level_zero_mode_pipeline_supported"])

    def test_positive_measure_summary_mismatch_fails_even_when_both_pass_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                positive_measure_fraction=0.5,
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertEqual(result["computed_path_evidence"]["positive_measure_fraction"], 1.0)
        self.assertFalse(
            result["computed_path_evidence"]["declared_positive_measure_matches_raw"]
        )
        self.assertFalse(result["gates"]["R6_raw_local_zero_mode_positive_measure"])

    def test_zero_same_meter_positive_control_fails_R7(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                same_meter_positive_control_cost=0.0,
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertFalse(result["gates"]["R7_same_meter_positive_control_nonzero"])

    def test_positive_control_summary_with_zero_raw_control_fails_R7(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                same_meter_positive_control_cost=1.0,
                same_meter_positive_control_costs=[
                    0.0 for _ in range(self.protocol["parameter_grid_points"])
                ],
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertEqual(
            result["computed_path_evidence"]["same_meter_positive_control_cost"],
            0.0,
        )
        self.assertFalse(result["gates"]["R7_same_meter_positive_control_nonzero"])

    def test_positive_control_summary_mismatch_fails_even_when_both_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(
                directory,
                same_meter_positive_control_cost=2.0,
            )
            cert = self._source_bound_certificate(directory, path_data)
            result = self._audit_with(directory, cert, path_data)
        self.assertEqual(
            result["computed_path_evidence"]["same_meter_positive_control_cost"],
            1.0,
        )
        self.assertFalse(
            result["computed_path_evidence"]["declared_control_cost_matches_raw"]
        )
        self.assertFalse(result["gates"]["R7_raw_same_meter_control_positive"])

    def test_target_G_constructed_witness_is_circular_negative_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            cert = self._source_bound_certificate(
                directory,
                path_data,
                witness_not_constructed_from_target_G=False,
            )
            result = self._audit_with(directory, cert, path_data)
        self.assertTrue(result["circular_negative_control"])
        self.assertFalse(result["gates"]["R8_witness_independent_of_target_G_TESC"])

    def test_valid_upstream_witness_does_not_certify_physical_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            path_data = self._write_path_data(directory)
            certificate_path = directory / "certificate.json"
            certificate_path.write_text(
                json.dumps(
                    self._source_bound_certificate_data(directory, path_data),
                    sort_keys=True,
                )
                + "\n"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_r_to_law1.py",
                    "--zero-mode-certificate",
                    str(certificate_path),
                    "--zero-mode-path-data",
                    str(path_data),
                    "--outdir",
                    str(directory / "out"),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        report = json.loads(completed.stdout)
        self.assertTrue(report["zero_mode_certificate_gate"])
        self.assertTrue(report["zero_mode_path_data_source_bound"])
        self.assertFalse(report["physical_zero_set_binding_certificate_gate"])
        self.assertFalse(report["R_plus_declared_structure_to_LawI_certified"])
        self.assertFalse(report["unconditional_R_alone_to_LawI_proved"])

    def test_nonconstant_zero_cost_path_gives_positive_measure_local_zero_mode(self) -> None:
        path = lambda t: np.array([t, 0.0])
        velocity = lambda _t: np.array([1.0, 0.0])
        cost = lambda _x, _v: 0.0
        record = accumulated_cost(
            cost,
            path,
            self.grid,
            protocol=self.protocol,
            velocity=velocity,
        )
        mask = np.linalg.norm(record["velocities"], axis=1) > self.protocol[
            "velocity_nonzero_tol"
        ]
        fraction = positive_measure_fraction(mask, self.grid)
        self.assertGreaterEqual(fraction, self.protocol["minimum_positive_measure_fraction"])
        self.assertAlmostEqual(record["accumulated_cost"], 0.0)

    def test_missing_protocol_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = dict(self.protocol)
            protocol.pop("protocol_sha256")
            path = Path(tmpdir, "protocol.json")
            path.write_text(json.dumps(protocol) + "\n")
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_existing_v011_lawi_regression_unchanged(self) -> None:
        protocol = load_frozen_protocol()
        G = derive_tesc_hessian(protocol)
        theorem = audit_conditional_theorem(G, protocol)
        self.assertLess(theorem["metrics"]["detG"], 0)
        self.assertTrue(theorem["conclusions"]["signature_must_be_1_1"])
        self.assertEqual(len(theorem["metrics"]["null_rays"]), 2)

    def test_load_zero_mode_certificate_rejects_malformed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "certificate.json")
            path.write_text(json.dumps({"schema_version": "unknown"}) + "\n")
            with self.assertRaises(ZeroModeCertificateError):
                load_zero_mode_certificate(path)


if __name__ == "__main__":
    unittest.main()
