from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r_to_law1.provenance import audit_provenance


class ProvenanceTests(unittest.TestCase):
    def test_physical_binding_and_native_unique_gates_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            source = base / "source.txt"
            source.write_text("source-bound statement\n")
            source_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
            certificate = {
                "nonnegative_realization_cost": {
                    "definition_source_path": "source.txt",
                    "definition_source_sha256": source_hash,
                    "predeclared": True,
                    "nonnegative": True,
                    "nonzero_zero_mode_attained": True,
                },
                "signed_zero_set_representative": {
                    "definition_source_path": "source.txt",
                    "definition_source_sha256": source_hash,
                    "real_C2": True,
                    "symmetric": True,
                    "nondegenerate": True,
                },
                "physical_zero_set_binding": {
                    "binding_source_path": "source.txt",
                    "binding_source_sha256": source_hash,
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
            path = base / "certificate.json"
            path.write_text(json.dumps(certificate) + "\n")
            result = audit_provenance(path)

        self.assertTrue(result["physical_zero_set_binding_provenance"]["gate"])
        self.assertFalse(result["native_unique_TESC_selection"]["gate"])

    def test_hash_field_alone_does_not_bind_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "certificate.json"
            certificate = {
                "nonnegative_realization_cost": {
                    "definition_source_path": "",
                    "definition_source_sha256": "a" * 64,
                    "predeclared": True,
                    "nonnegative": True,
                    "nonzero_zero_mode_attained": True,
                },
                "signed_zero_set_representative": {
                    "definition_source_path": "",
                    "definition_source_sha256": "a" * 64,
                    "real_C2": True,
                    "symmetric": True,
                    "nondegenerate": True,
                },
                "physical_zero_set_binding": {
                    "binding_source_path": "",
                    "binding_source_sha256": "a" * 64,
                    "Z_F_equals_Z_q_on_selected_plane": True,
                    "frozen_before_outcomes": True,
                },
            }
            path.write_text(json.dumps(certificate) + "\n")
            result = audit_provenance(path)

        self.assertFalse(result["physical_zero_set_binding_provenance"]["gate"])


if __name__ == "__main__":
    unittest.main()
