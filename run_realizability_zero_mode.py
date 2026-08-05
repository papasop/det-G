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
        return None
    return json.loads(source.read_text())


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
    certificate = (
        load_zero_mode_certificate(args.certificate) if args.certificate else None
    )
    path_record = load_path_record(args.path_data)
    certificate_path = Path(args.certificate) if args.certificate else None
    audit = audit_principle_r_witness(
        protocol,
        certificate,
        certificate_base_dir=certificate_path.parent if certificate_path else ".",
        path_record=path_record,
    )
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "Realizability zero-mode witness audit",
        "version": protocol["version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "certificate_supplied": certificate is not None,
        "path_data_supplied": path_record is not None,
        **audit,
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
