#!/usr/bin/env python3
"""Public entry point for the Principle-R to Lorentzian Law-I audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from r_to_law1.covariance import (  # noqa: E402
    audit_gl2_covariance,
    audit_lambda_sensitivity,
    audit_tesc_operational_inputs,
    audit_unit_rescaling,
)
from r_to_law1.finite_zero_set import trace_finite_zero_branches  # noqa: E402
from r_to_law1.provenance import audit_provenance  # noqa: E402
from r_to_law1.report import emit_report  # noqa: E402
from r_to_law1.tesc import derive_tesc_hessian, load_frozen_protocol  # noqa: E402
from r_to_law1.theorem import audit_conditional_theorem, construct_null_rays  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute the Principle-R to Lorentzian Law-I evidence package.")
    parser.add_argument("--protocol", default="protocols/frozen_tesc_protocol.json")
    parser.add_argument("--outdir", default="reference_results/v0.1.0")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook/kernel arguments:", unknown)
    protocol = load_frozen_protocol(args.protocol)
    provenance = audit_provenance(protocol["native_certificate"])
    physical_binding_gate = provenance["physical_zero_set_binding_provenance"]["gate"]

    G = derive_tesc_hessian(protocol)
    theorem = audit_conditional_theorem(G, protocol, physical_binding_gate=physical_binding_gate)
    theorem["metrics"].update(construct_null_rays(G))

    finite_branches = trace_finite_zero_branches(protocol)
    covariance = audit_gl2_covariance(protocol)
    units = audit_unit_rescaling(protocol)
    sensitivity = audit_lambda_sensitivity(protocol)
    operational_inputs = audit_tesc_operational_inputs(protocol)

    report = emit_report(
        protocol,
        theorem,
        finite_branches,
        covariance,
        units,
        sensitivity,
        operational_inputs,
        provenance,
        args.outdir,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
