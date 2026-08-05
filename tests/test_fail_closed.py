from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from audits.r_law1_two_channel_origin_audit_v3_1 import audit_cert, template


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


if __name__ == "__main__":
    unittest.main()
