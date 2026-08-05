from __future__ import annotations

import unittest

import numpy as np

from r_to_law1.theorem import audit_conditional_theorem


class ConditionalTheoremTests(unittest.TestCase):
    def test_binding_certificate_controls_premises_gate(self) -> None:
        protocol = {
            "principle_R": {"nonzero_zero_cost_direction_required": True},
            "structural_assumptions": {
                "principle_R_nonzero_direction_attained": True,
                "selected_process_space_dimension": 2,
                "signed_representative_is_real_C2": True,
                "stationary_basepoint_for_signed_representative": True,
                "signed_representative_is_symmetric": True,
                "signed_representative_is_nondegenerate": True,
                "nonnegative_realization_cost_predeclared": False,
                "physical_zero_set_equals_signed_representative_zero_set": False,
            },
            "decision_thresholds": {
                "nondegeneracy_tol": 1e-10,
                "hessian_discriminant_tol": 1e-10,
                "null_ray_residual_tol": 1e-10,
                "finite_root_residual_tol": 1e-9,
                "tangent_slope_relative_tol": 0.15,
                "gl2_covariance_relative_tol": 2e-5,
                "gl2_zero_set_tol": 1e-10,
                "tesc_conformal_residual_tol": 1e-8,
                "two_channel_independence_tol": 1e-12,
                "two_channel_null_residual_tol": 1e-12,
                "centering_value_tol": 1e-14,
                "centering_gradient_tol": 1e-8,
                "ray_equivalence_tol": 1e-12,
                "scale_closure_tol": 1e-12,
                "channel_independence_tol": 1e-12,
                "zero_set_forward_violation_tol": 0.0,
                "zero_set_reverse_violation_tol": 0.0,
                "minimum_branch_separation": 1e-6,
            },
        }
        G = np.array([[1.0, 0.0], [0.0, -1.0]])

        fail_closed = audit_conditional_theorem(G, protocol, physical_binding_gate=False)
        self.assertTrue(fail_closed["analytic_theorem_logic_gate"])
        self.assertFalse(fail_closed["conditional_theorem_premises_gate"])

        certified = audit_conditional_theorem(G, protocol, physical_binding_gate=True)
        self.assertTrue(certified["conditional_theorem_premises_gate"])


if __name__ == "__main__":
    unittest.main()
