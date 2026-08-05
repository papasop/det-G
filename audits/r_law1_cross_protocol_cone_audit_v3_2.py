#!/usr/bin/env python3
"""Prospective cross-protocol zero-cone naturality audit for R -> Law-I.

Manifest JSON schema is emitted on first run. Every protocol has its own CSV
with columns x,y,F,split. F must be independently defined and nonnegative.
The frozen map z=T x sends protocol coordinates to canonical coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from audits.common import canonical_hash, jsonable, sha256_file, source_hash_matches
except ModuleNotFoundError:
    from common import canonical_hash, jsonable, sha256_file, source_hash_matches


TITLE = "PRINCIPLE R -> LAW-I CROSS-PROTOCOL ZERO-CONE NATURALITY AUDIT"
VERSION = "3.2.1"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TEMPLATE = ROOT / "protocols" / "cross_protocol_manifest.template.json"
REQUIRED_MANIFEST_CRITERIA = {
    "F_zero_tol",
    "q_zero_tol",
    "minimum_zero_points",
    "maximum_violation_rate",
    "cone_residual_tol",
    "branch_angle_tol",
}


def manifest_criteria(manifest: dict[str, Any]) -> dict[str, float]:
    criteria = manifest.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("v3.2 manifest is missing versioned criteria")
    missing = sorted(REQUIRED_MANIFEST_CRITERIA - set(criteria))
    if missing:
        raise ValueError("v3.2 manifest criteria missing: " + ", ".join(missing))
    return {key: float(criteria[key]) for key in REQUIRED_MANIFEST_CRITERIA}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    required = {"x", "y", "F", "split"}
    if not data.dtype.names or not required.issubset(data.dtype.names):
        raise ValueError(f"{path}: need x,y,F,split")

    points = np.c_[data["x"].astype(float), data["y"].astype(float)]
    costs = data["F"].astype(float)
    splits = np.asarray(data["split"], dtype=str)

    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(costs)):
        raise ValueError(f"{path}: finite nonnegative F required")
    if np.any(costs < -1e-12):
        raise ValueError(f"{path}: finite nonnegative F required")
    if not {"train", "heldout"}.issubset(set(splits)):
        raise ValueError(f"{path}: train and heldout required")

    return points, np.maximum(costs, 0.0), splits


def normalized_quadratic_residual(points: np.ndarray, G: np.ndarray) -> np.ndarray:
    quadratic = np.einsum("ni,ij,nj->n", points, G, points)
    radius_squared = np.sum(points * points, axis=1)
    scale = np.maximum(np.linalg.norm(G) * radius_squared, 1e-300)
    return np.abs(quadratic) / scale


def fit_G(points: np.ndarray, zero_mask: np.ndarray) -> np.ndarray | None:
    zero_points = points[zero_mask]
    if len(zero_points) < 6:
        return None

    design = np.c_[
        zero_points[:, 0] ** 2,
        2.0 * zero_points[:, 0] * zero_points[:, 1],
        zero_points[:, 1] ** 2,
    ]
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    coefficients = vh[-1]
    G = np.array(
        [
            [coefficients[0], coefficients[1]],
            [coefficients[1], coefficients[2]],
        ]
    )
    norm = np.linalg.norm(G)
    if norm < 1e-15:
        return None
    return G / norm


def canonical_G(G: np.ndarray, transform: np.ndarray) -> np.ndarray:
    inverse_transform = np.linalg.inv(transform)
    candidate = inverse_transform.T @ G @ inverse_transform
    return candidate / np.linalg.norm(candidate)


def projective_residual(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, int, float]:
    # Null sets determine G only up to any nonzero scalar: compare both signs.
    positive_residual = np.linalg.norm(first - second)
    negative_residual = np.linalg.norm(first + second)
    if positive_residual <= negative_residual:
        return float(positive_residual), 1, float(positive_residual)
    return float(negative_residual), -1, float(positive_residual)


def cone_angles(G: np.ndarray) -> np.ndarray:
    theta = np.linspace(0.0, math.pi, 300000, endpoint=False)
    unit_circle = np.c_[np.cos(theta), np.sin(theta)]
    residuals = normalized_quadratic_residual(unit_circle, G)
    chosen: list[float] = []

    for index in np.argsort(residuals):
        angle = float(theta[index])
        separated = all(
            abs(((angle - existing + math.pi / 2.0) % math.pi) - math.pi / 2.0)
            > 1e-3
            for existing in chosen
        )
        if separated:
            chosen.append(angle)
        if len(chosen) == 2:
            break

    return np.sort(chosen)


def branch_distance(first: np.ndarray, second: np.ndarray) -> float:
    def unoriented_distance(a: float, b: float) -> float:
        return abs(((a - b + math.pi / 2.0) % math.pi) - math.pi / 2.0)

    direct = max(
        unoriented_distance(first[0], second[0]),
        unoriented_distance(first[1], second[1]),
    )
    swapped = max(
        unoriented_distance(first[0], second[1]),
        unoriented_distance(first[1], second[0]),
    )
    return float(min(direct, swapped))


def one_protocol(
    item: dict[str, Any],
    base_dir: Path,
    cfg: dict[str, float],
) -> dict[str, Any]:
    csv_path = (base_dir / item["data_csv"]).resolve()
    points, costs, splits = load_csv(csv_path)
    train_mask = splits == "train"
    heldout_mask = splits == "heldout"
    F_zero_mask = costs <= cfg["F_zero_tol"]

    G = fit_G(points, train_mask & F_zero_mask)
    transform = np.asarray(item["to_canonical_T"], dtype=float)
    if G is None:
        return {
            "name": item["name"],
            "gate": False,
            "error": "insufficient training F-zero geometry",
        }

    q_zero_mask = normalized_quadratic_residual(points, G) <= cfg["q_zero_tol"]
    heldout_Fzero_count = int(np.sum(heldout_mask & F_zero_mask))
    heldout_qzero_count = int(np.sum(heldout_mask & q_zero_mask))
    Fzero_qnonzero_count = int(
        np.sum(heldout_mask & F_zero_mask & ~q_zero_mask)
    )
    qzero_Fpositive_count = int(
        np.sum(heldout_mask & ~F_zero_mask & q_zero_mask)
    )

    eigenvalues = np.linalg.eigvalsh(G)
    det_G = float(np.linalg.det(G))
    canonical_candidate = canonical_G(G, transform)

    gates = {
        "data_hash_matches": sha256_file(csv_path) == item["data_sha256"],
        "cost_definition_source_bound": source_hash_matches(
            item,
            path_key="cost_definition_source_path",
            hash_key="cost_definition_source_sha256",
            base_dir=base_dir,
        ),
        "protocol_predeclared": bool(item["predeclared_before_cross_comparison"]),
        "mapping_predeclared": bool(item["mapping_predeclared_before_outcomes"]),
        "mapping_invertible": abs(float(np.linalg.det(transform))) > 1e-12,
        "heldout_Fzero_coverage": heldout_Fzero_count >= cfg["minimum_zero_points"],
        "heldout_qzero_coverage": heldout_qzero_count >= cfg["minimum_zero_points"],
        "Fzero_implies_qzero": Fzero_qnonzero_count
        / max(1, heldout_Fzero_count)
        <= cfg["maximum_violation_rate"],
        "qzero_implies_Fzero": qzero_Fpositive_count
        / max(1, heldout_qzero_count)
        <= cfg["maximum_violation_rate"],
        "fitted_G_Lorentzian": eigenvalues[0] < 0 < eigenvalues[1],
    }

    return {
        "name": item["name"],
        "data_sha256": sha256_file(csv_path),
        "G_protocol": G,
        "G_canonical": canonical_candidate,
        "eigenvalues": eigenvalues,
        "detG": det_G,
        "canonical_branch_angles": cone_angles(canonical_candidate),
        "heldout": {
            "Fzero": heldout_Fzero_count,
            "qzero": heldout_qzero_count,
            "Fzero_qnonzero": Fzero_qnonzero_count,
            "qzero_Fpositive": qzero_Fpositive_count,
        },
        "gates": gates,
        "gate": all(gates.values()),
    }


def audit_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    cfg: dict[str, float],
) -> dict[str, Any]:
    base_dir = manifest_path.parent
    records = [
        one_protocol(protocol, base_dir, cfg)
        for protocol in manifest["protocols"]
    ]
    valid_records = [record for record in records if "G_canonical" in record]

    pairs: list[dict[str, Any]] = []
    for first_index in range(len(valid_records)):
        for second_index in range(first_index + 1, len(valid_records)):
            first = valid_records[first_index]
            second = valid_records[second_index]
            residual, relative_sign, positive_residual = projective_residual(
                first["G_canonical"],
                second["G_canonical"],
            )
            branch_residual = branch_distance(
                first["canonical_branch_angles"],
                second["canonical_branch_angles"],
            )
            pairs.append(
                {
                    "pair": [first["name"], second["name"]],
                    "unoriented_projective_residual": residual,
                    "relative_sign": relative_sign,
                    "positive_conformal_residual": positive_residual,
                    "branch_pair_distance_radians": branch_residual,
                    "unoriented_cone_match": residual <= cfg["cone_residual_tol"]
                    and branch_residual <= cfg["branch_angle_tol"],
                    "positive_coorientation_match": positive_residual
                    <= cfg["cone_residual_tol"],
                }
            )

    provenance = {
        "at_least_two_protocols": len(records) >= 2,
        "all_protocol_costs_independently_defined": bool(
            manifest["provenance"]["costs_independent_of_each_other_and_TESC"]
        ),
        "comparison_rule_frozen": bool(
            manifest["provenance"]["comparison_rule_frozen_before_outcomes"]
        ),
        "no_outcome_based_protocol_selection": bool(
            manifest["provenance"]["no_protocol_selected_after_outcomes"]
        ),
    }
    gates = {
        "provenance": all(provenance.values()),
        "all_protocols_pass_local_binding": len(records) >= 2
        and all(record["gate"] for record in records),
        "all_pairs_same_unoriented_cone": bool(pairs)
        and all(pair["unoriented_cone_match"] for pair in pairs),
    }
    oriented = gates["all_pairs_same_unoriented_cone"] and all(
        pair["positive_coorientation_match"] for pair in pairs
    )
    return {
        "protocol_records": records,
        "pairwise_naturality": pairs,
        "provenance_gates": provenance,
        "gates": gates,
        "cross_protocol_unoriented_cone_class_supported": all(gates.values()),
        "positive_coorientation_selected": bool(oriented),
        "gate": all(gates.values()),
    }


def synth(outdir: Path, cfg: dict[str, float]) -> dict[str, bool]:
    rng = np.random.default_rng(20260805)
    G = np.array([[1.0, 0.0], [0.0, -1.0]])
    source_path = outdir / "selftest_cost_definition.txt"
    source_path.write_text("synthetic cost definition for v3.2 self-test\n")
    source_hash = sha256_file(source_path)

    items: list[dict[str, Any]] = []
    transforms = (
        np.eye(2),
        np.array([[1.4, 0.3], [-0.2, 0.9]]),
    )
    for index, transform in enumerate(transforms):
        rows = ["x,y,F,split"]
        inverse_transform = np.linalg.inv(transform)
        for sample_index in range(800):
            canonical_point = rng.normal(size=2)
            canonical_point /= np.linalg.norm(canonical_point)
            protocol_point = inverse_transform @ canonical_point
            quadratic = float(canonical_point @ G @ canonical_point)
            cost = abs(quadratic)
            split = "train" if sample_index % 2 == 0 else "heldout"
            rows.append(
                f"{protocol_point[0]:.17g},{protocol_point[1]:.17g},{cost:.17g},{split}"
            )

        csv_path = outdir / f"selftest_protocol_{index}.csv"
        csv_path.write_text("\n".join(rows) + "\n")
        items.append(
            {
                "name": f"p{index}",
                "data_csv": csv_path.name,
                "data_sha256": sha256_file(csv_path),
                "cost_definition_source_path": source_path.name,
                "cost_definition_source_sha256": source_hash,
                "predeclared_before_cross_comparison": True,
                "mapping_predeclared_before_outcomes": True,
                "to_canonical_T": transform.tolist(),
            }
        )

    manifest = {
        "schema": "r-law1-cross-protocol-v3.2.1-selftest",
        "criteria": cfg,
        "protocols": items,
        "provenance": {
            "costs_independent_of_each_other_and_TESC": True,
            "comparison_rule_frozen_before_outcomes": True,
            "no_protocol_selected_after_outcomes": True,
        },
    }

    # Add exact ray points. The purely random samples rarely hit F <= tolerance.
    for item in items:
        csv_path = outdir / item["data_csv"]
        transform = np.asarray(item["to_canonical_T"])
        inverse_transform = np.linalg.inv(transform)
        lines = csv_path.read_text().splitlines()

        for sample_index in range(80):
            sign = 1.0 if sample_index % 2 == 0 else -1.0
            canonical_point = np.array([1.0, sign]) * rng.uniform(0.1, 1.0)
            protocol_point = inverse_transform @ canonical_point
            split = "train" if sample_index % 4 < 2 else "heldout"
            lines.append(
                f"{protocol_point[0]:.17g},{protocol_point[1]:.17g},0,{split}"
            )

        csv_path.write_text("\n".join(lines) + "\n")
        item["data_sha256"] = sha256_file(csv_path)

    positive_control = audit_manifest(manifest, outdir / "selftest_manifest.json", cfg)

    wrong_mapping_manifest = json.loads(json.dumps(manifest))
    wrong_mapping_manifest["protocols"][1]["to_canonical_T"] = [[1, 0], [0, 1]]
    negative_control = audit_manifest(
        wrong_mapping_manifest,
        outdir / "selftest_bad_manifest.json",
        cfg,
    )

    return {
        "matched_cross_protocol_positive_control_pass": positive_control["gate"],
        "wrong_mapping_negative_control_rejected": not negative_control["gate"],
    }


def template_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_TEMPLATE.read_text())


def write_template(outdir: Path) -> dict[str, Any]:
    for name in ("protocol_A.csv", "protocol_B.csv"):
        (outdir / name).write_text("x,y,F,split\n0.0,0.0,nan,train\n")

    manifest = template_manifest()
    (outdir / "cross_protocol_manifest_template.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--outdir", default="r_law1_cross_protocol_v3_2_results")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook/kernel arguments:", unknown)

    started_at = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest_sha256 = sha256_file(manifest_path)
        cfg = manifest_criteria(manifest)
        controls = synth(outdir, cfg)
        empirical_audit = audit_manifest(manifest, manifest_path, cfg)
        status = (
            "CROSS_PROTOCOL_UNORIENTED_LAWI_CONE_CLASS_SUPPORTED"
            if empirical_audit["gate"]
            else "CROSS_PROTOCOL_CONE_NATURALITY_NOT_SUPPORTED_FAIL_CLOSED"
        )
        locked_protocol = manifest
    else:
        locked_protocol = write_template(outdir)
        cfg = manifest_criteria(locked_protocol)
        controls = synth(outdir, cfg)
        empirical_audit = None
        manifest_sha256 = None
        status = "PIPELINE_CALIBRATED_INDEPENDENT_PROTOCOL_DATA_REQUIRED"

    locked_protocol = dict(locked_protocol)
    locked_protocol["manifest_sha256"] = manifest_sha256
    locked_protocol["locked_protocol_sha256"] = canonical_hash(locked_protocol)

    report = {
        "title": TITLE,
        "version": VERSION,
        "scientific_status": status,
        "protocol_sha256": locked_protocol["locked_protocol_sha256"],
        "manifest_supplied": empirical_audit is not None,
        "manifest_sha256": manifest_sha256,
        "self_tests": controls,
        "empirical_audit": empirical_audit,
        "all_scientific_gates_pass": bool(
            empirical_audit and empirical_audit["gate"] and all(controls.values())
        ),
        "interpretation": (
            "Each protocol must independently recover a held-out two-branch zero "
            "cone. Frozen realization maps must carry the unordered branch pair "
            "to one common projective conformal class. Zero sets alone determine "
            "G only up to nonzero scale; positive coorientation is reported "
            "separately."
        ),
        "next_required_step": (
            "Populate two independently sourced protocol CSVs and freeze their "
            "coordinate maps before comparing outcomes; then rerun with --manifest."
        ),
        "claim_boundary": (
            "A pass supports a local cross-protocol unoriented cone class, not a "
            "physical time orientation, metric scale, spacetime, Law-II/III or "
            "wavefunction."
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

    print("=" * 112)
    print(f"{TITLE} v{VERSION}")
    print("=" * 112)
    print(json.dumps(jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    return_code = main()
    if not any(name in sys.modules for name in ("ipykernel", "IPython", "google.colab")):
        raise SystemExit(return_code)
