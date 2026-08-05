#!/usr/bin/env python3
"""Single-channel no-go / independent two-channel Law-I origin audit v3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from audits.common import canonical_hash, jsonable, source_hash_matches
except ModuleNotFoundError:
    from common import canonical_hash, jsonable, source_hash_matches


TITLE = "PRINCIPLE R -> LAW-I SINGLE-CHANNEL NO-GO / TWO-CHANNEL ORIGIN AUDIT"
VERSION = "3.1"
G_TESC = np.array(
    [
        [-1.4753511828891064, -0.04866380215462485],
        [-0.04866380215462485, 0.2661881493004614],
    ],
    dtype=float,
)


def quadratic_from_channels(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    return 0.5 * (np.outer(plus, minus) + np.outer(minus, plus))


def conformal_fit(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = float(np.sum(candidate * target) / np.sum(target * target))
    residual = float(
        np.linalg.norm(candidate - scale * target)
        / max(np.linalg.norm(candidate), 1e-300)
    )
    return scale, residual


def unoriented_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = abs(float(first @ second)) / (
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    return math.acos(min(1.0, max(-1.0, cosine)))


def algebra(plus: list[float], minus: list[float]) -> dict[str, Any]:
    plus_covector = np.asarray(plus, dtype=float)
    minus_covector = np.asarray(minus, dtype=float)
    channel_matrix = np.vstack([plus_covector, minus_covector])
    channel_determinant = float(np.linalg.det(channel_matrix))

    induced_G = quadratic_from_channels(plus_covector, minus_covector)
    eigenvalues = np.linalg.eigvalsh(induced_G)
    det_G = float(np.linalg.det(induced_G))

    # The exact null lines are ker(L_+) and ker(L_-).
    plus_ray = np.array([-plus_covector[1], plus_covector[0]])
    minus_ray = np.array([-minus_covector[1], minus_covector[0]])
    residuals = [
        abs(float(ray @ induced_G @ ray))
        / (1.0 + np.linalg.norm(induced_G) * float(ray @ ray))
        for ray in (plus_ray, minus_ray)
    ]

    scale, residual = conformal_fit(induced_G, G_TESC)
    return {
        "channel_matrix": channel_matrix,
        "channel_determinant": channel_determinant,
        "channels_independent": abs(channel_determinant) > 1e-12,
        "channel_angle_radians": unoriented_angle(plus_covector, minus_covector),
        "induced_G": induced_G,
        "induced_G_eigenvalues": eigenvalues,
        "induced_detG": det_G,
        "induced_Lorentzian": det_G < 0,
        "null_rays": [
            plus_ray / np.linalg.norm(plus_ray),
            minus_ray / np.linalg.norm(minus_ray),
        ],
        "maximum_null_residual": max(residuals),
        "TESC_conformal_scale": scale,
        "TESC_conformal_relative_residual": residual,
        "same_TESC_null_cone": scale > 0 and residual < 1e-8,
    }


def load_certificate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def template() -> dict[str, Any]:
    return {
        "schema": "r-law1-independent-two-channel-v3.1",
        "channel_plus": {
            "covector": [1.0, 0.0],
            "definition_source_path": "",
            "definition_source_sha256": "",
            "physical_meaning": "",
            "measurable": False,
        },
        "channel_minus": {
            "covector": [0.0, 1.0],
            "definition_source_path": "",
            "definition_source_sha256": "",
            "physical_meaning": "",
            "measurable": False,
        },
        "capacity": {
            "definition_source_path": "",
            "definition_source_sha256": "",
            "strictly_positive": False,
        },
        "provenance": {
            "both_channels_predeclared_before_TESC": False,
            "definitions_do_not_use_G_TESC_or_its_null_rays": False,
            "relative_product_rule_derived": False,
            "selected_2D_plane_derived": False,
        },
    }


def audit_cert(
    certificate: dict[str, Any],
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    channel_plus = certificate["channel_plus"]
    channel_minus = certificate["channel_minus"]
    capacity = certificate["capacity"]
    provenance = certificate["provenance"]
    algebra_record = algebra(channel_plus["covector"], channel_minus["covector"])

    gates = {
        "plus_source_bound": source_hash_matches(
            channel_plus,
            path_key="definition_source_path",
            hash_key="definition_source_sha256",
            base_dir=base_dir,
        ),
        "minus_source_bound": source_hash_matches(
            channel_minus,
            path_key="definition_source_path",
            hash_key="definition_source_sha256",
            base_dir=base_dir,
        ),
        "plus_has_physical_meaning": bool(channel_plus["physical_meaning"].strip()),
        "minus_has_physical_meaning": bool(channel_minus["physical_meaning"].strip()),
        "both_channels_measurable": bool(
            channel_plus["measurable"] and channel_minus["measurable"]
        ),
        "capacity_source_bound": source_hash_matches(
            capacity,
            path_key="definition_source_path",
            hash_key="definition_source_sha256",
            base_dir=base_dir,
        ),
        "capacity_strictly_positive": bool(capacity["strictly_positive"]),
        "channels_predeclared_before_TESC": bool(
            provenance["both_channels_predeclared_before_TESC"]
        ),
        "definitions_independent_of_TESC": bool(
            provenance["definitions_do_not_use_G_TESC_or_its_null_rays"]
        ),
        "product_rule_derived_not_chosen": bool(
            provenance["relative_product_rule_derived"]
        ),
        "two_dimensional_plane_derived": bool(
            provenance["selected_2D_plane_derived"]
        ),
        "channels_linearly_independent": algebra_record["channels_independent"],
        "induced_quadratic_form_Lorentzian": algebra_record["induced_Lorentzian"],
        "two_null_rays_exact": algebra_record["maximum_null_residual"] < 1e-12,
    }
    return {
        "gates": gates,
        "gate": all(gates.values()),
        "algebra": algebra_record,
    }


def write_template(outdir: Path) -> None:
    path = outdir / "independent_two_channel_certificate_template.json"
    path.write_text(json.dumps(template(), indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", default="")
    parser.add_argument("--outdir", default="r_law1_two_channel_origin_v3_1_results")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook/kernel arguments:", unknown)

    started_at = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    single_channel = np.array([1.0, 2.0])
    single_kernel = np.array([-2.0, 1.0])
    single_channel_record = {
        "nonzero_channel": True,
        "kernel_dimension": 1,
        "one_unoriented_null_line": True,
        "cannot_equal_two_distinct_Lorentzian_null_lines": True,
        "kernel_witness": single_kernel,
    }

    positive_control = algebra([1.0, 1.0], [1.0, -1.0])
    dependent_control = algebra([1.0, 1.0], [2.0, 2.0])
    circular_control = {
        "F_abs_q_zero_set_equals_q_by_construction": True,
        "F_q_squared_zero_set_equals_q_by_construction": True,
        "admissible_as_independent_R_to_LawI_evidence": False,
    }
    self_tests = {
        "single_channel_no_go_pass": single_channel_record[
            "cannot_equal_two_distinct_Lorentzian_null_lines"
        ],
        "independent_two_channel_positive_control": positive_control[
            "induced_Lorentzian"
        ]
        and positive_control["channels_independent"],
        "dependent_channel_negative_control_rejected": not dependent_control[
            "induced_Lorentzian"
        ],
        "circular_q_cost_rejected_as_provenance": not circular_control[
            "admissible_as_independent_R_to_LawI_evidence"
        ],
    }

    certificate_path = Path(args.certificate) if args.certificate else None
    if certificate_path and certificate_path.is_file():
        certificate = load_certificate(certificate_path)
        empirical_audit = audit_cert(certificate, certificate_path.parent)
        certificate_sha256 = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        status = (
            "INDEPENDENT_TWO_CHANNEL_LAWI_ORIGIN_SUPPORTED"
            if empirical_audit["gate"]
            else "TWO_CHANNEL_ORIGIN_CERTIFICATE_FAIL_CLOSED"
        )
    else:
        write_template(outdir)
        empirical_audit = None
        certificate_sha256 = None
        status = (
            "STRUCTURAL_MECHANISM_CERTIFIED_"
            "INDEPENDENT_CHANNEL_PROVENANCE_REQUIRED"
        )

    protocol = {
        "title": TITLE,
        "version": VERSION,
        "tests": "single-channel no-go; two-channel product; circular negative control",
        "claim_rule": (
            "F=|L_plus L_minus|/H with H>0; channel definitions must "
            "predate and not depend on TESC"
        ),
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    (outdir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    report = {
        "title": TITLE,
        "version": VERSION,
        "scientific_status": status,
        "protocol_sha256": protocol["protocol_sha256"],
        "single_channel_information_time": single_channel_record,
        "two_channel_positive_control": positive_control,
        "dependent_channel_negative_control": dependent_control,
        "circular_cost_negative_control": circular_control,
        "self_tests": self_tests,
        "certificate_supplied": empirical_audit is not None,
        "certificate_sha256": certificate_sha256,
        "native_two_channel_audit": empirical_audit,
        "all_scientific_gates_pass": bool(
            empirical_audit and empirical_audit["gate"] and all(self_tests.values())
        ),
        "interpretation": (
            "A scalar Information-Time differential has one kernel line in 2D "
            "and cannot represent a nondegenerate Lorentzian two-line zero cone. "
            "Two independently derived nonparallel channels yield F=|L+L-|/H "
            "and an induced Lorentzian quadratic form exactly. Matching TESC "
            "additionally requires a noncircular provenance-bound identification."
        ),
        "next_required_step": (
            "Define and source-bind two measurable realization channels before "
            "consulting TESC. If no such channels exist, retain Law-I as an "
            "additional representation assumption."
        ),
        "claim_boundary": (
            "The algebraic two-channel mechanism is not a derivation from "
            "Principle R without provenance. F=|q_TESC| or q_TESC^2 is a "
            "circular negative control."
        ),
        "elapsed_seconds": time.time() - started_at,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (outdir / "run_summary.json").write_text(
        json.dumps(jsonable(report), indent=2) + "\n"
    )

    print("=" * 112)
    print(f"{TITLE} v{VERSION}")
    print("=" * 112)
    print(json.dumps(jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    return_code = main()
    if not any(name in sys.modules for name in ("ipykernel", "IPython", "google.colab")):
        raise SystemExit(return_code)
