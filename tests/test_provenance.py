from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r_to_law1.provenance import audit_provenance


HASH = "a" * 64


class ProvenanceTests(unittest.TestCase):
    def test_physical_binding_and_native_unique_gates_are_separate(self) -> None:
        certificate = {
            "nonnegative_realization_cost": {
                "definition_source_sha256": HASH,
                "predeclared": True,
                "nonnegative": True,
                "nonzero_zero_mode_attained": True,
            },
            "signed_zero_set_representative": {
                "definition_source_sha256": HASH,
                "real_C2": True,
                "symmetric": True,
                "nondegenerate": True,
            },
            "physical_zero_set_binding": {
                "binding_source_sha256": HASH,
                "Z_F_equals_Z_q_on_selected_plane": True,
                "frozen_before_outcomes": True,
            },
            "native_unique_TESC_selection": {
                "task_minus_exposure_derived": False,
                "relative_negative_sign_derived": False,
                "lambda_physical_normalization_derived": False,
                "two_dimensional_process_plane_derived": False,
                "zero_set_completeness_derived": False,
                "TESC_equivalence_class_unique": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "certificate.json"
            path.write_text(json.dumps(certificate) + "\n")
            result = audit_provenance(path)

        self.assertTrue(result["physical_zero_set_binding_provenance"]["gate"])
        self.assertFalse(result["native_unique_TESC_selection"]["gate"])


if __name__ == "__main__":
    unittest.main()
