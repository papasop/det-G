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
from r_to_law1.protocol import threshold  # noqa: E402
from realizability.certificate import load_zero_mode_certificate  # noqa: E402
from realizability.protocol import load_realizability_protocol  # noqa: E402
from realizability.zero_mode import audit_principle_r_witness  # noqa: E402


def load_json_file(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = json.loads(source.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def fail_closed_zero_mode(
    error: str,
    *,
    certificate_supplied: bool,
    path_data_supplied: bool,
) -> dict:
    return {
        "certificate_supplied": certificate_supplied,
        "path_data_supplied": path_data_supplied,
        "principle_R_witness_source_bound": False,
        "path_data_source_bound": False,
        "principle_R_witness_certified": False,
        "errors": [error],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the Principle-R to Lorentzian Law-I evidence package."
    )
    parser.add_argument("--protocol", default="protocols/frozen_tesc_protocol.json")
    parser.add_argument(
        "--certificate",
        default=None,
        help="Override the provenance certificate path declared by the protocol.",
    )
    parser.add_argument("--outdir", default="reference_results/v0.1.1")
    parser.add_argument(
        "--zero-mode-protocol",
        default="protocols/frozen_realizability_protocol.json",
    )
    parser.add_argument("--zero-mode-certificate", default="")
    parser.add_argument("--zero-mode-path-data", default="")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook/kernel arguments:", unknown)
    protocol = load_frozen_protocol(args.protocol)
    if args.certificate is not None:
        protocol = dict(protocol)
        protocol["native_certificate"] = args.certificate
    provenance = audit_provenance(protocol["native_certificate"])
    if args.zero_mode_certificate or args.zero_mode_path_data:
        if not args.zero_mode_certificate or not args.zero_mode_path_data:
            zero_mode = fail_closed_zero_mode(
                "zero-mode certificate and path data must be supplied together",
                certificate_supplied=bool(args.zero_mode_certificate),
                path_data_supplied=bool(args.zero_mode_path_data),
            )
        else:
            try:
                zero_protocol = load_realizability_protocol(args.zero_mode_protocol)
                zero_certificate = load_zero_mode_certificate(args.zero_mode_certificate)
                zero_path_record = load_json_file(args.zero_mode_path_data)
                zero_mode = audit_principle_r_witness(
                    zero_protocol,
                    zero_certificate,
                    certificate_base_dir=Path(args.zero_mode_certificate).parent,
                    path_record=zero_path_record,
                    path_record_source_path=args.zero_mode_path_data,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                zero_mode = fail_closed_zero_mode(
                    str(exc),
                    certificate_supplied=bool(args.zero_mode_certificate),
                    path_data_supplied=bool(args.zero_mode_path_data),
                )
    else:
        zero_mode = {
            "certificate_supplied": False,
            "path_data_supplied": False,
            "principle_R_witness_source_bound": False,
            "path_data_source_bound": False,
            "principle_R_witness_certified": False,
        }
    physical_binding_gate = provenance["physical_zero_set_binding_provenance"]["gate"]

    G = derive_tesc_hessian(protocol)
    theorem = audit_conditional_theorem(G, protocol, physical_binding_gate=physical_binding_gate)
    theorem["metrics"].update(
        construct_null_rays(G, tolerance=threshold(protocol, "hessian_discriminant_tol"))
    )

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
        zero_mode,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
