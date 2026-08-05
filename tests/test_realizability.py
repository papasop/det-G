from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from realizability.certificate import ZeroModeCertificate
from realizability.path_cost import (
    accumulated_cost,
    path_is_nonconstant,
    positive_measure_fraction,
)
from realizability.protocol import RealizabilityProtocolError, load_realizability_protocol
from realizability.zero_mode import audit_principle_r_witness
from r_to_law1.tesc import derive_tesc_hessian, load_frozen_protocol
from r_to_law1.theorem import audit_conditional_theorem


class RealizabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_realizability_protocol()
        self.grid = np.linspace(0.0, 1.0, 11)

    def _source_bound_certificate(
        self,
        directory: Path,
        **overrides,
    ) -> ZeroModeCertificate:
        files = {}
        for name in ("cost", "path", "admissible", "vacuum"):
            source = directory / f"{name}.txt"
            source.write_text(f"{name} source\n")
            files[name] = (
                source.name,
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
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
        return ZeroModeCertificate.from_dict(data)

    def test_constant_path_rejected(self) -> None:
        path = lambda _t: np.array([1.0, 0.0])
        self.assertFalse(path_is_nonconstant(path, self.grid, 1e-10))

    def test_nonconstant_positive_cost_path_is_not_zero_witness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cert = self._source_bound_certificate(Path(tmpdir))
            record = {
                "finite_cost_values": True,
                "cost_nonnegative": True,
                "accumulated_cost": 1.0,
                "positive_measure_fraction": 1.0,
                "same_meter_positive_control_cost": 1.0,
            }
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=tmpdir,
                path_record=record,
            )
        self.assertFalse(result["gates"]["R5_accumulated_cost_zero"])
        self.assertFalse(result["principle_R_witness_certified"])

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
            "path_nonconstant_tol"
        ]
        fraction = positive_measure_fraction(mask, self.grid)
        self.assertGreaterEqual(fraction, self.protocol["minimum_positive_measure_fraction"])
        self.assertAlmostEqual(record["accumulated_cost"], 0.0)

    def test_measure_zero_nonzero_velocity_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cert = self._source_bound_certificate(Path(tmpdir))
            record = {
                "finite_cost_values": True,
                "cost_nonnegative": True,
                "accumulated_cost": 0.0,
                "positive_measure_fraction": 0.0,
                "same_meter_positive_control_cost": 1.0,
            }
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=tmpdir,
                path_record=record,
            )
        self.assertFalse(result["gates"]["R6_local_zero_mode_positive_measure"])

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
            cert = self._source_bound_certificate(Path(tmpdir))
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=tmpdir,
                path_record={
                    **record,
                    "positive_measure_fraction": 1.0,
                    "same_meter_positive_control_cost": 1.0,
                },
            )
        self.assertFalse(result["gates"]["R2_protocol_and_nonnegative_cost_predeclared"])

    def test_zero_same_meter_positive_control_fails_R7(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cert = self._source_bound_certificate(Path(tmpdir))
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=tmpdir,
                path_record={
                    "finite_cost_values": True,
                    "cost_nonnegative": True,
                    "accumulated_cost": 0.0,
                    "positive_measure_fraction": 1.0,
                    "same_meter_positive_control_cost": 0.0,
                },
            )
        self.assertFalse(result["gates"]["R7_same_meter_positive_control_nonzero"])

    def test_missing_source_file_fails_closed(self) -> None:
        cert = ZeroModeCertificate.from_dict(
            {
                "protocol_sha256": self.protocol["protocol_sha256"],
                "cost_source_path": "missing.txt",
                "cost_source_sha256": "a" * 64,
            }
        )
        result = audit_principle_r_witness(self.protocol, cert)
        self.assertFalse(result["principle_R_witness_source_bound"])

    def test_forged_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cert = self._source_bound_certificate(
                Path(tmpdir),
                cost_source_sha256="a" * 64,
            )
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=tmpdir,
            )
        self.assertFalse(result["principle_R_witness_source_bound"])

    def test_missing_protocol_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = dict(self.protocol)
            protocol.pop("protocol_sha256")
            path = Path(tmpdir, "protocol.json")
            path.write_text(json.dumps(protocol) + "\n")
            with self.assertRaises(RealizabilityProtocolError):
                load_realizability_protocol(path)

    def test_target_G_constructed_witness_is_circular_negative_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cert = self._source_bound_certificate(
                Path(tmpdir),
                witness_not_constructed_from_target_G=False,
            )
            result = audit_principle_r_witness(
                self.protocol,
                cert,
                certificate_base_dir=tmpdir,
            )
        self.assertTrue(result["circular_negative_control"])
        self.assertFalse(result["gates"]["R8_witness_independent_of_target_G_TESC"])

    def test_no_certificate_keeps_existing_lawI_conditional(self) -> None:
        protocol = load_frozen_protocol()
        G = derive_tesc_hessian(protocol)
        theorem = audit_conditional_theorem(G, protocol, physical_binding_gate=False)
        self.assertTrue(theorem["analytic_theorem_logic_gate"])
        self.assertFalse(theorem["conditional_theorem_premises_gate"])

    def test_existing_v011_lawi_regression_unchanged(self) -> None:
        protocol = load_frozen_protocol()
        G = derive_tesc_hessian(protocol)
        theorem = audit_conditional_theorem(G, protocol)
        self.assertLess(theorem["metrics"]["detG"], 0)
        self.assertTrue(theorem["conclusions"]["signature_must_be_1_1"])
        self.assertEqual(len(theorem["metrics"]["null_rays"]), 2)


if __name__ == "__main__":
    unittest.main()
