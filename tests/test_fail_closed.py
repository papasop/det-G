from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from audits.r_law1_two_channel_origin_audit_v3_1 import audit_cert, template
from r_to_law1.covariance import audit_lambda_sensitivity
from r_to_law1.protocol import ProtocolError
from r_to_law1.tesc import load_frozen_protocol
from r_to_law1.theorem import audit_conditional_theorem


class FailClosedAuditTests(unittest.TestCase):
    def test_two_channel_template_fails_closed(self) -> None:
        result = audit_cert(template())
        self.assertFalse(result["gate"])
        self.assertFalse(result["gates"]["plus_source_bound"])
        self.assertTrue(result["gates"]["channels_linearly_independent"])

    def test_bidirectional_binding_without_data_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    "audits/r_law1_bidirectional_zero_set_audit_v3_0.py",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads(Path(tmpdir, "run_summary.json").read_text())

        self.assertFalse(summary["data_supplied"])
        self.assertFalse(summary["all_scientific_gates_pass"])
        self.assertTrue(all(summary["self_tests"].values()))

    def test_cross_protocol_without_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    "audits/r_law1_cross_protocol_cone_audit_v3_2.py",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads(Path(tmpdir, "run_summary.json").read_text())

        self.assertFalse(summary["manifest_supplied"])
        self.assertFalse(summary["all_scientific_gates_pass"])
        self.assertTrue(all(summary["self_tests"].values()))

    def test_protocol_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path("protocols/frozen_tesc_protocol.json")
            protocol = json.loads(source.read_text())
            protocol["tesc_lambda"] = 1.25
            path = Path(tmpdir, "bad_protocol.json")
            path.write_text(json.dumps(protocol) + "\n")

            with self.assertRaises(ProtocolError):
                load_frozen_protocol(path)

    def test_missing_protocol_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path("protocols/frozen_tesc_protocol.json")
            protocol = json.loads(source.read_text())
            protocol.pop("protocol_sha256")
            path = Path(tmpdir, "missing_sha_protocol.json")
            path.write_text(json.dumps(protocol) + "\n")

            with self.assertRaises(ProtocolError):
                load_frozen_protocol(path)

    def test_missing_gate_threshold_fails_closed(self) -> None:
        protocol = json.loads(Path("protocols/frozen_tesc_protocol.json").read_text())
        protocol.pop("protocol_sha256", None)
        protocol["decision_thresholds"].pop("null_ray_residual_tol")

        with self.assertRaises(ProtocolError):
            audit_conditional_theorem([[1.0, 0.0], [0.0, -1.0]], protocol)

    def test_lambda_window_is_not_grid_density_artifact(self) -> None:
        protocol = load_frozen_protocol()
        sparse = json.loads(json.dumps(protocol))
        sparse["covariance"]["lambda_grid"] = [1.0]

        dense_result = audit_lambda_sensitivity(protocol)
        sparse_result = audit_lambda_sensitivity(sparse)

        self.assertTrue(
            dense_result["gates"]["analytic_open_Lorentzian_interval_around_one"]
        )
        self.assertTrue(
            sparse_result["gates"]["analytic_open_Lorentzian_interval_around_one"]
        )
        self.assertTrue(dense_result["gates"]["Lorentzian_not_unique_to_lambda_one"])
        self.assertTrue(sparse_result["gates"]["Lorentzian_not_unique_to_lambda_one"])


if __name__ == "__main__":
    unittest.main()
