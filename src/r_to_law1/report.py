"""Aggregate and emit the public Principle-R to Law-I report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def emit_report(
    protocol: dict[str, Any],
    theorem: dict[str, Any],
    finite_branches: dict[str, Any],
    covariance: dict[str, Any],
    units: dict[str, Any],
    sensitivity: dict[str, Any],
    operational_inputs: dict[str, Any],
    native_selection: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signed_tesc_gates = {
        "G_real_symmetric": theorem["declared_structural_premises"]["quadratic_form_symmetric"],
        "G_nondegenerate": theorem["declared_structural_premises"]["quadratic_form_nondegenerate"],
        "detG_negative": theorem["conclusions"]["detG_must_be_negative"],
        "signature_1_1": theorem["conclusions"]["signature_must_be_1_1"],
        "two_distinct_null_rays_constructed": theorem["conclusions"]["null_set_is_two_distinct_real_rays"],
        "null_ray_residuals_pass": theorem["metrics"]["maximum_null_ray_residual"] < 1e-10,
        "finite_two_branch_zero_contrast_set_observed": finite_branches["gates"]["all_sections_have_two_roots"],
        "finite_zero_contrast_residual_pass": finite_branches["gates"]["all_root_residuals_small"],
        "no_extra_zero_contrast_branch_in_frozen_domain": finite_branches["gates"]["no_extra_zero_branches_to_maximum_boundary"],
        "signed_zero_set_GL2_covariance_pass": covariance["gates"]["GL2_finite_zero_set_covariance"],
        "unit_rescaling_signature_preserved": units["gates"]["signature_preserved_under_extreme_unit_rescaling"],
        "lambda_sensitivity_pass": sensitivity["gates"]["Lorentzian_not_unique_to_lambda_one"],
        "operational_inputs_pass": operational_inputs["gate"],
    }
    signed_tesc_gate = all(signed_tesc_gates.values())
    native_gate = bool(native_selection["gate"])
    physical_binding_gate = theorem["physical_zero_set_binding_certificate_gate"]
    conditional_support_gate = theorem["conditional_theorem_premises_gate"] and signed_tesc_gate
    report = {
        "title": "Principle R to Lorentzian Law I",
        "version": protocol["version"],
        "scientific_status": (
            "UNCONDITIONAL_R_TO_LAW_I_NATIVE_DERIVATION_CERTIFIED"
            if conditional_support_gate and native_gate
            else "SIGNED_TESC_ZERO_SET_REPRESENTATIVE_SUPPORTED_PHYSICAL_ZERO_SET_BINDING_OPEN"
            if theorem["analytic_theorem_logic_gate"] and signed_tesc_gate
            else "R_TO_LAW_I_PREMISES_OR_SIGNED_REPRESENTATIVE_INCOMPLETE_FAIL_CLOSED"
        ),
        "logical_statement": "R plus a certified binding Z(F)∩V = Z(q)∩V, with q(v)=v^T G v real symmetric nondegenerate on dim(V)=2, implies det(G)<0, signature (1,1), two null rays",
        "theorem_proof_kind": "analytic_linear_algebra",
        "theorem_numerically_proved": False,
        "analytic_theorem_logic_gate": theorem["analytic_theorem_logic_gate"],
        "conditional_theorem": {
            "declared_structural_premises": theorem["declared_structural_premises"],
            "physical_zero_set_binding": theorem["physical_zero_set_binding"],
            "conclusions": theorem["conclusions"],
            "analytic_theorem_logic_gate": theorem["analytic_theorem_logic_gate"],
            "declared_structural_premises_gate": theorem["declared_structural_premises_gate"],
            "physical_zero_set_binding_certificate_gate": physical_binding_gate,
            "premises_gate": theorem["conditional_theorem_premises_gate"],
        },
        "conditional_theorem_premises_gate": theorem["conditional_theorem_premises_gate"],
        "signed_TESC_zero_set_representative": {
            "gates": signed_tesc_gates,
            "gate": signed_tesc_gate,
        },
        "physical_zero_set_binding_certificate_gate": physical_binding_gate,
        "native_R_selection": native_selection,
        "metrics": {
            **theorem["metrics"],
            **finite_branches["metrics"],
            **covariance["metrics"],
            **sensitivity["metrics"],
        },
        "signed_TESC_zero_set_representative_supported": signed_tesc_gate,
        "conditional_R_to_LawI_supported": conditional_support_gate,
        "unconditional_R_alone_to_LawI_proved": conditional_support_gate and native_gate,
        "all_scientific_gates_pass": conditional_support_gate and native_gate,
        "interpretation": "The exact result is conditional: R plus a certified zero-set binding between the nonnegative realization cost F and an independently declared signed quadratic representative q forces Lorentzian Law I. TESC supplies a frozen signed zero-set representative. It does not show that F=q, D^2F=G_TESC, or that R alone uniquely selects TESC.",
        "next_required_step": "supply a source-bound certificate for the physical zero-set binding Z(F)∩V = Z(q)∩V, and derive task-minus-centred-exposure, its relative normalization, two-dimensionality and global completeness from Principle R",
        "claim_boundary": "No Law II/III, spacetime metric, (1,3) signature, physical light cone, wavefunction, Born rule, Cloud or QPU claim.",
        "artifacts": {
            "run_summary": str(output / "run_summary.json"),
            "finite_zero_branches": str(output / "finite_zero_branches.json"),
            "covariance_records": str(output / "covariance_records.json"),
        },
    }
    report = to_jsonable(report)
    (output / "finite_zero_branches.json").write_text(json.dumps(to_jsonable(finite_branches["records"]), indent=2) + "\n")
    (output / "covariance_records.json").write_text(json.dumps(to_jsonable(covariance["records"]), indent=2) + "\n")
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
