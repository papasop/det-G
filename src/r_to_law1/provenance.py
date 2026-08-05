"""Native Principle-R selection gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_native_r_selection(certificate_path: str | Path) -> dict[str, Any]:
    path = Path(certificate_path)
    certificate = json.loads(path.read_text())
    nonnegative_cost = certificate["nonnegative_realization_cost"]
    signed_representative = certificate["signed_zero_set_representative"]
    zero_set_binding = certificate["physical_zero_set_binding"]

    def has_sha256(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64

    gates = {
        "nonnegative_realization_cost_source_bound": has_sha256(
            nonnegative_cost["definition_source_sha256"]
        ),
        "nonnegative_realization_cost_predeclared": bool(nonnegative_cost["predeclared"]),
        "nonnegative_realization_cost_nonnegative": bool(nonnegative_cost["nonnegative"]),
        "nonnegative_zero_mode_attained": bool(nonnegative_cost["nonzero_zero_mode_attained"]),
        "signed_zero_set_representative_source_bound": has_sha256(
            signed_representative["definition_source_sha256"]
        ),
        "signed_zero_set_representative_real_C2": bool(signed_representative["real_C2"]),
        "signed_zero_set_representative_symmetric": bool(signed_representative["symmetric"]),
        "signed_zero_set_representative_nondegenerate": bool(
            signed_representative["nondegenerate"]
        ),
        "physical_zero_set_binding_source_bound": has_sha256(
            zero_set_binding["binding_source_sha256"]
        ),
        "Z_F_equals_Z_q_on_selected_plane": bool(
            zero_set_binding["Z_F_equals_Z_q_on_selected_plane"]
        ),
        "binding_frozen_before_outcomes": bool(zero_set_binding["frozen_before_outcomes"]),
    }
    return {"gates": gates, "gate": all(gates.values()), "certificate_path": str(path)}
