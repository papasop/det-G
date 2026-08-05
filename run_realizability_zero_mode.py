#!/usr/bin/env python3
"""Public entry for path-level Principle-R zero-mode witness audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from r_to_law1.protocol import jsonable  # noqa: E402
from realizability.certificate import load_zero_mode_certificate  # noqa: E402
from realizability.protocol import load_realizability_protocol  # noqa: E402
from realizability.zero_mode import audit_principle_r_witness  # noqa: E402


def load_path_record(path: str | None) -> dict | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"path-data file not found: {path}")
    return json.loads(source.read_text())


def fail_closed_audit(
    error: str,
    *,
    certificate_supplied: bool,
    path_data_supplied: bool,
) -> dict:
    gates = {
        "R1_state_admissible_domain_source_bound": False,
        "R2_protocol_and_nonnegative_cost_predeclared": False,
        "R3_contraction_family_gives_zero_infimum": False,
        "R4_attained_path_finite_and_nonconstant": False,
        "R5_accumulated_cost_zero": False,
        "R6_local_zero_mode_positive_measure": False,
        "R7_same_meter_positive_control_nonzero": False,
        "R8_witness_independent_of_target_G_TESC": False,
        "R9_path_data_source_bound": False,
    }
    return {
        "gates": gates,
        "certificate_supplied": certificate_supplied,
        "path_data_supplied": path_data_supplied,
        "path_level_R_pipeline_supported": False,
        "zero_infimum_certified": False,
        "attained_nonconstant_zero_cost_path": False,
        "local_zero_mode_positive_measure": False,
        "principle_R_witness_source_bound": False,
        "path_data_source_bound": False,
        "principle_R_witness_certified": False,
        "self_tests": {},
        "circular_negative_control": False,
        "errors": [error],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="protocols/frozen_realizability_protocol.json")
    parser.add_argument("--certificate", default="")
    parser.add_argument("--path-data", default="")
    parser.add_argument("--outdir", default="reference_results/realizability_v0.1.0")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook/kernel arguments:", unknown)

    protocol = load_realizability_protocol(args.protocol)
    certificate = None
    path_record = None
    certificate_path = Path(args.certificate) if args.certificate else None
    path_data_path = Path(args.path_data) if args.path_data else None
    error = ""
    if bool(args.certificate) != bool(args.path_data):
        error = "zero-mode certificate and path data must be supplied together"
    else:
        try:
            certificate = (
                load_zero_mode_certificate(args.certificate) if args.certificate else None
            )
            path_record = load_path_record(args.path_data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)

    if error:
        audit = fail_closed_audit(
            error,
            certificate_supplied=bool(args.certificate),
            path_data_supplied=bool(args.path_data),
        )
    else:
        audit = audit_principle_r_witness(
            protocol,
            certificate,
            certificate_base_dir=certificate_path.parent if certificate_path else ".",
            path_record=path_record,
            path_record_source_path=path_data_path,
        )
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "Realizability zero-mode witness audit",
        "version": protocol["version"],
        "protocol_sha256": protocol["protocol_sha256"],
        **audit,
        "all_zero_mode_certificate_gates_pass": bool(
            audit["principle_R_witness_certified"]
        ),
        "universal_Principle_R_proved": False,
        "all_scientific_gates_pass": False,
        "claim_boundary": (
            "This path-level interface does not prove Principle R as a universal "
            "law and does not prove physical TESC binding, spacetime, Law-II/III "
            "or wavefunction claims."
        ),
    }
    (output / "protocol.json").write_text(json.dumps(jsonable(protocol), indent=2) + "\n")
    (output / "run_summary.json").write_text(json.dumps(jsonable(report), indent=2) + "\n")
    (output / "zero_mode_records.json").write_text(
        json.dumps(jsonable({"path_record": path_record}), indent=2) + "\n"
    )
    print(json.dumps(jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
