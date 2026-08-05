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
    tesc_witness_gates = {
        "G_real_symmetric": theorem["premises"]["quadratic_form_symmetric"],
        "G_nondegenerate": theorem["premises"]["quadratic_form_nondegenerate"],
        "detG_negative": theorem["conclusions"]["detG_must_be_negative"],
        "signature_1_1": theorem["conclusions"]["signature_must_be_1_1"],
        "two_distinct_null_rays_constructed": theorem["conclusions"]["null_set_is_two_distinct_real_rays"],
        "null_ray_residuals_pass": theorem["metrics"]["maximum_null_ray_residual"] < 1e-10,
        "finite_two_branch_zero_set_observed": finite_branches["gates"]["all_sections_have_two_roots"],
        "finite_zero_residual_pass": finite_branches["gates"]["all_root_residuals_small"],
        "no_extra_branch_in_frozen_domain": finite_branches["gates"]["no_extra_zero_branches_to_maximum_boundary"],
        "zero_set_GL2_covariance_pass": covariance["gates"]["GL2_finite_zero_set_covariance"],
        "unit_rescaling_signature_preserved": units["gates"]["signature_preserved_under_extreme_unit_rescaling"],
        "lambda_sensitivity_pass": sensitivity["gates"]["Lorentzian_not_unique_to_lambda_one"],
        "operational_inputs_pass": operational_inputs["gate"],
    }
    tesc_witness_gate = all(tesc_witness_gates.values())
    native_gate = bool(native_selection["gate"])
    report = {
        "title": "Principle R to Lorentzian Law I",
        "version": protocol["version"],
        "scientific_status": (
            "UNCONDITIONAL_R_TO_LAW_I_NATIVE_DERIVATION_CERTIFIED"
            if theorem["gate"] and tesc_witness_gate and native_gate
            else "CONDITIONAL_R_PLUS_STRUCTURE_IMPLIES_LAW_I_TESC_WITNESS_SUPPORTED_NATIVE_SELECTION_OPEN"
            if theorem["gate"] and tesc_witness_gate
            else "R_TO_LAW_I_PREMISES_OR_WITNESS_INCOMPLETE_FAIL_CLOSED"
        ),
        "logical_statement": "Principle R + A_2D,C2,stationary,complete,symmetric,nondegenerate => det(G)<0, signature (1,1), two null rays",
        "exact_conditional_theorem": {
            "premises": theorem["premises"],
            "conclusions": theorem["conclusions"],
            "gate": theorem["gate"],
        },
        "operational_TESC_witness": {
            "gates": tesc_witness_gates,
            "gate": tesc_witness_gate,
        },
        "native_R_selection": native_selection,
        "metrics": {
            **theorem["metrics"],
            **finite_branches["metrics"],
            **covariance["metrics"],
            **sensitivity["metrics"],
        },
        "conditional_R_to_LawI_supported": theorem["gate"] and tesc_witness_gate,
        "unconditional_R_alone_to_LawI_proved": theorem["gate"] and tesc_witness_gate and native_gate,
        "all_scientific_gates_pass": theorem["gate"] and tesc_witness_gate and native_gate,
        "interpretation": "The exact result is conditional: R plus the declared 2D quadratic-completeness assumptions forces Lorentzian Law I. TESC supplies a frozen operational existence witness. It does not show that R alone uniquely selects TESC or proves global completeness.",
        "next_required_step": "derive task-minus-centred-exposure, its relative normalization, two-dimensionality and global zero-set completeness from a source-bound Principle-R construction",
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
