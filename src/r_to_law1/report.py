"""Aggregate and emit the public Principle-R to Law-I report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import threshold


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
    provenance: dict[str, Any],
    output_dir: str | Path,
    zero_mode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    null_ray_tol = threshold(protocol, "null_ray_residual_tol")
    signed_tesc_gates = {
        "G_real_symmetric": theorem["declared_structural_premises"]["quadratic_form_symmetric"],
        "G_nondegenerate": theorem["declared_structural_premises"]["quadratic_form_nondegenerate"],
        "detG_negative": theorem["conclusions"]["detG_must_be_negative"],
        "signature_1_1": theorem["conclusions"]["signature_must_be_1_1"],
        "two_distinct_null_rays_constructed": theorem["conclusions"]["null_set_is_two_distinct_real_rays"],
        "null_ray_residuals_pass": theorem["metrics"]["maximum_null_ray_residual"] < null_ray_tol,
        "finite_two_branch_zero_contrast_set_observed": finite_branches["gates"]["all_sections_have_two_roots"],
        "finite_zero_contrast_residual_pass": finite_branches["gates"]["all_root_residuals_small"],
        "no_extra_zero_contrast_branch_in_frozen_domain": finite_branches["gates"]["no_extra_zero_branches_to_maximum_boundary"],
        "signed_zero_set_GL2_pullback_identity_pass": covariance["gates"][
            "GL2_zero_set_pullback_identity"
        ],
        "unit_rescaling_signature_preserved": units["gates"]["signature_preserved_under_extreme_unit_rescaling"],
        "lambda_sensitivity_pass": sensitivity["gates"]["Lorentzian_not_unique_to_lambda_one"],
        "operational_inputs_pass": operational_inputs["gate"],
    }
    signed_tesc_gate = all(signed_tesc_gates.values())
    zero_mode = zero_mode or {
        "certificate_supplied": False,
        "principle_R_witness_source_bound": False,
        "principle_R_witness_certified": False,
    }
    physical_binding_provenance = provenance["physical_zero_set_binding_provenance"]
    native_unique_selection = provenance["native_unique_TESC_selection"]
    native_unique_gate = bool(native_unique_selection["gate"])
    physical_binding_gate = theorem["physical_zero_set_binding_certificate_gate"]
    conditional_support_gate = theorem["conditional_theorem_premises_gate"] and signed_tesc_gate
    declared_structure_certified = (
        theorem["conditional_theorem_premises_gate"]
        and signed_tesc_gate
        and theorem["declared_structural_premises_gate"]
        and bool(zero_mode["principle_R_witness_certified"])
    )
    if declared_structure_certified and native_unique_gate:
        scientific_status = "R_PLUS_DECLARED_STRUCTURE_TO_LAW_I_CERTIFIED"
    elif conditional_support_gate:
        scientific_status = "CONDITIONAL_R_TO_LAWI_SUPPORTED_NATIVE_TESC_SELECTION_OPEN"
    elif theorem["analytic_theorem_logic_gate"] and signed_tesc_gate:
        scientific_status = "SIGNED_TESC_ZERO_SET_REPRESENTATIVE_SUPPORTED_PHYSICAL_ZERO_SET_BINDING_OPEN"
    else:
        scientific_status = "R_TO_LAW_I_PREMISES_OR_SIGNED_REPRESENTATIVE_INCOMPLETE_FAIL_CLOSED"

    report = {
        "title": "Principle R to Lorentzian Law I",
        "version": protocol["version"],
        "protocol_schema": protocol["schema"],
        "protocol_sha256": protocol.get("protocol_sha256"),
        "derivation_version": protocol.get("derivation_version"),
        "scientific_status": scientific_status,
        "logical_statement": "R plus a certified binding Z(F)∩V = Z(q)∩V, with q(v)=v^T G v real symmetric nondegenerate on dim(V)=2, implies det(G)<0, signature (1,1), two null rays",
        "theorem_proof_kind": "analytic_linear_algebra",
        "theorem_numerically_proved": False,
        "analytic_theorem_logic_gate": theorem["analytic_theorem_logic_gate"],
        "conditional_theorem": {
            "declared_structural_premises": theorem["declared_structural_premises"],
            "physical_zero_set_binding": physical_binding_provenance,
            "protocol_zero_set_binding_declaration": theorem["physical_zero_set_binding"],
            "conclusions": theorem["conclusions"],
            "analytic_theorem_logic_gate": theorem["analytic_theorem_logic_gate"],
            "declared_structural_premises_gate": theorem["declared_structural_premises_gate"],
            "physical_zero_set_binding_certificate_gate": physical_binding_gate,
            "protocol_zero_set_binding_declaration_gate": theorem[
                "protocol_zero_set_binding_declaration_gate"
            ],
            "premises_gate": theorem["conditional_theorem_premises_gate"],
        },
        "conditional_theorem_premises_gate": theorem["conditional_theorem_premises_gate"],
        "principle_R_local_zero_mode_assumed": theorem[
            "declared_structural_premises"
        ]["principle_R_local_zero_mode_adopted"],
        "zero_mode_certificate_supplied": bool(zero_mode["certificate_supplied"]),
        "zero_mode_certificate_source_bound": bool(
            zero_mode["principle_R_witness_source_bound"]
        ),
        "zero_mode_certificate_gate": bool(
            zero_mode["principle_R_witness_certified"]
        ),
        "signed_TESC_zero_set_representative": {
            "gates": signed_tesc_gates,
            "gate": signed_tesc_gate,
        },
        "physical_zero_set_binding_certificate_gate": physical_binding_gate,
        "physical_zero_set_binding_provenance": physical_binding_provenance,
        "native_unique_TESC_selection": native_unique_selection,
        "native_unique_TESC_selection_gate": native_unique_gate,
        "metrics": {
            **theorem["metrics"],
            **finite_branches["metrics"],
            **covariance["metrics"],
            **sensitivity["metrics"],
        },
        "signed_TESC_zero_set_representative_supported": signed_tesc_gate,
        "conditional_R_to_LawI_supported": conditional_support_gate,
        "R_plus_declared_structure_to_LawI_certified": declared_structure_certified
        and native_unique_gate,
        "unconditional_R_alone_to_LawI_proved": False,
        "all_scientific_gates_pass": False,
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
    (output / "finite_zero_branches.json").write_text(
        json.dumps(to_jsonable(finite_branches["records"]), indent=2) + "\n"
    )
    (output / "covariance_records.json").write_text(
        json.dumps(to_jsonable(covariance["records"]), indent=2) + "\n"
    )
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
