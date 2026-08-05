"""Source-bound provenance gates for zero-set binding and native selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_provenance(certificate_path: str | Path) -> dict[str, Any]:
    path = Path(certificate_path)
    certificate = json.loads(path.read_text())
    nonnegative_cost = certificate["nonnegative_realization_cost"]
    signed_representative = certificate["signed_zero_set_representative"]
    zero_set_binding = certificate["physical_zero_set_binding"]
    unique_selection = certificate.get("native_unique_TESC_selection", {})

    def has_sha256(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64

    physical_binding_gates = {
        "nonnegative_F_definition_source_bound": has_sha256(
            nonnegative_cost["definition_source_sha256"]
        ),
        "nonnegative_F_predeclared": bool(nonnegative_cost["predeclared"]),
        "nonnegative_F_nonnegative": bool(nonnegative_cost["nonnegative"]),
        "nonnegative_zero_mode_attained": bool(nonnegative_cost["nonzero_zero_mode_attained"]),
        "signed_q_definition_source_bound": has_sha256(
            signed_representative["definition_source_sha256"]
        ),
        "signed_q_real_C2": bool(signed_representative["real_C2"]),
        "signed_q_symmetric": bool(signed_representative["symmetric"]),
        "signed_q_nondegenerate": bool(signed_representative["nondegenerate"]),
        "physical_zero_set_binding_source_bound": has_sha256(
            zero_set_binding["binding_source_sha256"]
        ),
        "Z_F_equals_Z_q_on_selected_plane": bool(
            zero_set_binding["Z_F_equals_Z_q_on_selected_plane"]
        ),
        "binding_frozen_before_outcomes": bool(zero_set_binding["frozen_before_outcomes"]),
    }
    native_unique_gates = {
        "task_minus_exposure_derived": bool(unique_selection.get("task_minus_exposure_derived")),
        "relative_negative_sign_derived": bool(unique_selection.get("relative_negative_sign_derived")),
        "lambda_physical_normalization_derived": bool(
            unique_selection.get("lambda_physical_normalization_derived")
        ),
        "two_dimensional_process_plane_derived": bool(
            unique_selection.get("two_dimensional_process_plane_derived")
        ),
        "zero_set_completeness_derived": bool(unique_selection.get("zero_set_completeness_derived")),
        "TESC_equivalence_class_unique": bool(unique_selection.get("TESC_equivalence_class_unique")),
    }
    return {
        "certificate_path": str(path),
        "physical_zero_set_binding_provenance": {
            "gates": physical_binding_gates,
            "gate": all(physical_binding_gates.values()),
        },
        "native_unique_TESC_selection": {
            "gates": native_unique_gates,
            "gate": all(native_unique_gates.values()),
        },
    }
