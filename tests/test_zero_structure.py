from __future__ import annotations

import unittest

import numpy as np

from audits.rc_zero_structure_to_law1_audit_v0_1 import run_audit
from r_to_law1.channel_origin import (
    analytic_two_channel_determinant_identity,
    audit_two_channel_origin,
)
from r_to_law1.tesc import load_frozen_protocol
from r_to_law1.zero_set_binding import audit_zero_set_binding_certificate
from r_to_law1.zero_structure import (
    normalize_unoriented_ray,
    zero_structure_from_single_channel,
    zero_structure_from_two_channels,
)


class ZeroStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_frozen_protocol()

    def test_single_channel_classifies_as_single_line(self) -> None:
        structure = zero_structure_from_single_channel([1.0, 2.0], self.protocol)
        self.assertEqual(structure.zero_set_kind, "single_line")

    def test_independent_two_channel_classifies_as_two_branch_cone(self) -> None:
        structure = zero_structure_from_two_channels(
            [1.0, 1.0],
            [1.0, -1.0],
            self.protocol,
        )
        self.assertEqual(structure.zero_set_kind, "two_branch_cone")

    def test_swapping_channels_does_not_change_structure(self) -> None:
        first = zero_structure_from_two_channels([1.0, 1.0], [1.0, -1.0], self.protocol)
        second = zero_structure_from_two_channels([1.0, -1.0], [1.0, 1.0], self.protocol)
        self.assertEqual(first.branch_directions, second.branch_directions)

    def test_nonzero_scalar_does_not_change_unoriented_ray(self) -> None:
        self.assertEqual(
            normalize_unoriented_ray([1.0, -2.0], self.protocol),
            normalize_unoriented_ray([-3.0, 6.0], self.protocol),
        )

    def test_dependent_two_channel_degenerates_and_rejects_lawi(self) -> None:
        structure = zero_structure_from_two_channels([1.0, 1.0], [2.0, 2.0], self.protocol)
        audit = audit_two_channel_origin([1.0, 1.0], [2.0, 2.0], self.protocol)
        self.assertEqual(structure.zero_set_kind, "single_line")
        self.assertFalse(audit["gate"])

    def test_determinant_identity_exact(self) -> None:
        identity = analytic_two_channel_determinant_identity(
            [1.0, 1.0],
            [1.0, -1.0],
            self.protocol,
        )
        self.assertAlmostEqual(identity["detG"], identity["expected_detG"])
        self.assertTrue(identity["determinant_identity_verified"])

    def test_forward_only_binding_is_not_equivalence(self) -> None:
        result = audit_zero_set_binding_certificate(
            {
                "forward_inclusion": True,
                "reverse_inclusion": False,
            }
        )
        self.assertFalse(result["zero_set_equivalence_certified"])

    def test_reverse_only_binding_is_not_equivalence(self) -> None:
        result = audit_zero_set_binding_certificate(
            {
                "forward_inclusion": False,
                "reverse_inclusion": True,
            }
        )
        self.assertFalse(result["zero_set_equivalence_certified"])

    def test_circular_constructions_are_negative_controls(self) -> None:
        for construction_kind in ("abs_q", "q_g_squared"):
            result = audit_zero_set_binding_certificate(
                {
                    "construction_kind": construction_kind,
                    "forward_inclusion": True,
                    "reverse_inclusion": True,
                    "sources_independent": True,
                }
            )
            self.assertTrue(result["circular_negative_control"])
            self.assertFalse(result["zero_set_equivalence_certified"])

    def test_unified_rc_audit_fails_closed_without_certificate(self) -> None:
        result = run_audit(self.protocol, None, None)
        self.assertFalse(result["physical_zero_set_binding_certified"])
        self.assertFalse(result["unconditional_R_to_LawI_proved"])
        self.assertFalse(result["all_scientific_gates_pass"])

    def test_reference_law_i_result_remains_unchanged(self) -> None:
        from r_to_law1.tesc import derive_tesc_hessian
        from r_to_law1.theorem import audit_conditional_theorem

        G = derive_tesc_hessian(self.protocol)
        theorem = audit_conditional_theorem(G, self.protocol)
        self.assertLess(theorem["metrics"]["detG"], 0)
        self.assertTrue(theorem["conclusions"]["signature_must_be_1_1"])
        self.assertEqual(len(theorem["metrics"]["null_rays"]), 2)

    def test_zero_vector_is_not_branch(self) -> None:
        with self.assertRaises(ValueError):
            normalize_unoriented_ray(np.zeros(2), self.protocol)


if __name__ == "__main__":
    unittest.main()
