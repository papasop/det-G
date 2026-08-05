"""Native Principle-R selection gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_native_r_selection(certificate_path: str | Path) -> dict[str, Any]:
    path = Path(certificate_path)
    certificate = json.loads(path.read_text())
    native = certificate["native_derivation"]
    hashes_bound = all(
        isinstance(native[key], str) and len(native[key]) == 64
        for key in (
            "principle_R_source_sha256",
            "cost_definition_source_sha256",
            "relative_coefficient_source_sha256",
        )
    )
    gates = {
        "three_source_hashes_bound": hashes_bound,
        "definitions_frozen_before_outcomes": bool(native["definitions_frozen_before_outcomes"]),
        "R_derives_task_minus_exposure": bool(native["R_derives_task_minus_exposure"]),
        "R_derives_relative_coefficient": bool(native["R_derives_relative_coefficient"]),
        "R_proves_global_zero_set_completeness": bool(native["R_proves_global_zero_set_completeness"]),
        "R_uniquely_selects_this_cost": bool(native["R_uniquely_selects_this_cost"]),
    }
    return {"gates": gates, "gate": all(gates.values()), "certificate_path": str(path)}
