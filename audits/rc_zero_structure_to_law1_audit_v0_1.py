#!/usr/bin/env python3
"""RC zero-structure to conditional Law-I preflight audit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from audits.common import canonical_hash, jsonable
except ModuleNotFoundError:
    from common import canonical_hash, jsonable

from r_to_law1.channel_origin import (
    audit_single_channel_no_go,
    audit_two_channel_origin,
)
from r_to_law1.tesc import derive_tesc_hessian, load_frozen_protocol
from r_to_law1.theorem import audit_conditional_theorem
from r_to_law1.zero_set_binding import audit_zero_set_binding_certificate


TITLE = "RC ZERO-STRUCTURE TO CONDITIONAL LAW-I AUDIT"
VERSION = "0.1"


def load_certificate(path: str) -> tuple[dict[str, Any] | None, Path | None]:
    if not path:
        return None, None
    certificate_path = Path(path)
    if not certificate_path.is_file():
        return None, None
    return json.loads(certificate_path.read_text()), certificate_path


def run_audit(
    protocol: dict[str, Any],
    binding_certificate: dict[str, Any] | None,
    certificate_path: Path | None,
) -> dict[str, Any]:
    single = audit_single_channel_no_go([1.0, 2.0], protocol)
    two_channel = audit_two_channel_origin([1.0, 1.0], [1.0, -1.0], protocol)
    binding = audit_zero_set_binding_certificate(
        binding_certificate,
        base_dir=certificate_path.parent if certificate_path else ".",
        protocol=protocol,
    )
    G = derive_tesc_hessian(protocol)
    law_i = audit_conditional_theorem(
        G,
        protocol,
        physical_binding_gate=binding["zero_set_equivalence_certified"],
    )
    gate_1 = {
        "F_nonnegative": bool(binding_certificate),
        "F_origin_zero": bool(binding_certificate),
        "cost_source_predeclared": binding["gates"]["physical_cost_source_bound"],
    }
    gate_2 = {
        "nontrivial_zero_structure": two_channel[
            "zero_structure"
        ].zero_set_kind == "two_branch_cone",
        "positive_scale_closed": two_channel[
            "zero_structure"
        ].positive_scale_closed,
    }
    gate_3 = {
        "two_distinct_unoriented_zero_rays": two_channel[
            "two_branch_zero_cone_supported"
        ],
        "not_single_kernel_line": single["single_channel_no_go_certified"],
    }
    gate_4 = binding["gates"]
    gate_5 = {
        "G_symmetric": law_i["declared_structural_premises"][
            "quadratic_form_symmetric"
        ],
        "G_nondegenerate": law_i["declared_structural_premises"][
            "quadratic_form_nondegenerate"
        ],
        "detG_negative": law_i["conclusions"]["detG_must_be_negative"],
        "signature_1_1": law_i["conclusions"]["signature_must_be_1_1"],
    }
    physical_binding = binding["zero_set_equivalence_certified"]
    law_i_conditional = bool(all(gate_5.values()) and physical_binding)
    return {
        "gates": {
            "Gate_1_RC_cost": gate_1,
            "Gate_2_nontrivial_zero_structure": gate_2,
            "Gate_3_two_branch_structure": gate_3,
            "Gate_4_bidirectional_binding": gate_4,
            "Gate_5_LawI_representation": gate_5,
        },
        "RC_zero_structure_supported": all(gate_1.values())
        and all(gate_2.values()),
        "single_channel_no_go_certified": single["single_channel_no_go_certified"],
        "two_channel_complexity_required": True,
        "two_branch_zero_cone_supported": two_channel[
            "two_branch_zero_cone_supported"
        ],
        "physical_zero_set_binding_certified": physical_binding,
        "LawI_representation_conditional": law_i_conditional,
        "unconditional_R_to_LawI_proved": False,
        "all_scientific_gates_pass": False,
        "single_channel_audit": single,
        "two_channel_audit": two_channel,
        "zero_set_binding_audit": binding,
        "LawI_audit": law_i,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="protocols/frozen_tesc_protocol.json")
    parser.add_argument("--binding-certificate", default="")
    parser.add_argument("--outdir", default="rc_zero_structure_to_law1_results")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook/kernel arguments:", unknown)

    started_at = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    protocol = load_frozen_protocol(args.protocol)
    certificate, certificate_path = load_certificate(args.binding_certificate)
    audit = run_audit(protocol, certificate, certificate_path)
    locked_protocol = {
        "title": TITLE,
        "version": VERSION,
        "frozen_protocol_sha256": protocol["protocol_sha256"],
        "certificate_supplied": certificate is not None,
    }
    locked_protocol["protocol_sha256"] = canonical_hash(locked_protocol)
    report = {
        "title": TITLE,
        "version": VERSION,
        "scientific_status": (
            "RC_ZERO_STRUCTURE_INTERFACE_FAIL_CLOSED_PHYSICAL_BINDING_OPEN"
        ),
        "protocol_sha256": locked_protocol["protocol_sha256"],
        **audit,
        "claim_boundary": (
            "This is an interface and falsification audit, not new physical "
            "evidence. It does not prove native channels, unique TESC, "
            "spacetime, Law-II/III or a wavefunction."
        ),
        "elapsed_seconds": time.time() - started_at,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (outdir / "protocol.json").write_text(
        json.dumps(jsonable(locked_protocol), indent=2) + "\n"
    )
    (outdir / "run_summary.json").write_text(
        json.dumps(jsonable(report), indent=2) + "\n"
    )
    print(json.dumps(jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
