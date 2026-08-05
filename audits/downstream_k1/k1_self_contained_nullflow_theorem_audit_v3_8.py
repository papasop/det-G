#!/usr/bin/env python3
"""Downstream K=1 kernel-image null-flow theorem audit.

This is a downstream auxiliary audit. It checks the self-contained
two-dimensional algebraic theorem that, for a Lorentzian symmetric
quadratic representative G, the Law-III critical factor

    B_c = J G - d_c I

has two native G-null lines given by ker(B_c) and Im(B_c).

It does not certify that the Principle-R realization-cost zero set equals
these two lines, and it does not derive the Law-III critical selection.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r_to_law1.protocol import jsonable, protocol_sha256, threshold
from r_to_law1.tesc import derive_tesc_hessian, load_frozen_protocol

J = np.array([[0.0, -1.0], [1.0, 0.0]])
IDENTITY = np.eye(2)


def unoriented_ray(vector: np.ndarray, tolerance: float) -> np.ndarray:
    """Return a deterministic representative of an unoriented real ray."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("zero vector cannot represent a null-flow line")
    ray = vector / norm
    pivot = int(np.argmax(np.abs(ray)))
    if ray[pivot] < 0:
        ray = -ray
    return ray


def kernel_line(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    _, singular_values, right_vectors = np.linalg.svd(matrix)
    if float(singular_values[-1]) > tolerance:
        raise ValueError("matrix has no numerical kernel at the declared tolerance")
    return unoriented_ray(right_vectors[-1], tolerance)


def image_line(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    _, singular_values, _ = np.linalg.svd(matrix)
    rank = int(np.sum(singular_values > tolerance))
    if rank != 1:
        raise ValueError("matrix image is not one-dimensional")
    for column in matrix.T:
        if np.linalg.norm(column) > tolerance:
            return unoriented_ray(column, tolerance)
    raise ValueError("matrix image has no nonzero generator")


def ray_separation(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    dot = min(1.0, max(0.0, dot))
    return float(math.sqrt(max(0.0, 1.0 - dot * dot)))


def quadratic_residual(G: np.ndarray, ray: np.ndarray) -> float:
    return abs(float(ray.T @ G @ ray))


def audit_kernel_image_nullflow(
    G: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    nondegeneracy_tol = threshold(protocol, "nondegeneracy_tol")
    null_tol = threshold(protocol, "null_ray_residual_tol")
    separation_tol = threshold(protocol, "minimum_branch_separation")

    G = np.asarray(G, dtype=float)
    detG = float(np.linalg.det(G))
    symmetric = bool(np.allclose(G, G.T, atol=null_tol, rtol=0.0))
    lorentzian = bool(symmetric and detG < -nondegeneracy_tol)

    if lorentzian:
        critical_damping = math.sqrt(-detG)
        Bc = J @ G - critical_damping * IDENTITY
        rank = int(np.linalg.matrix_rank(Bc, tol=nondegeneracy_tol))
        kernel = kernel_line(Bc, nondegeneracy_tol)
        image = image_line(Bc, nondegeneracy_tol)
        kernel_residual = quadratic_residual(G, kernel)
        image_residual = quadratic_residual(G, image)
        separation = ray_separation(kernel, image)
    else:
        critical_damping = None
        Bc = np.full((2, 2), np.nan)
        rank = 0
        kernel = np.full(2, np.nan)
        image = np.full(2, np.nan)
        kernel_residual = math.inf
        image_residual = math.inf
        separation = 0.0

    gates = {
        "G_real_symmetric": symmetric,
        "G_Lorentzian_det_negative": lorentzian,
        "critical_factor_rank_one": rank == 1,
        "kernel_line_G_null": kernel_residual <= null_tol,
        "image_line_G_null": image_residual <= null_tol,
        "kernel_and_image_distinct": separation > separation_tol,
    }
    theorem_gate = all(gates.values())

    return {
        "critical_damping": critical_damping,
        "J": J,
        "B_c": Bc,
        "kernel_line": kernel,
        "image_line": image,
        "kernel_G_null_residual": kernel_residual,
        "image_G_null_residual": image_residual,
        "kernel_image_projective_separation": separation,
        "gates": gates,
        "native_K1_auxiliary_two_line_theorem_certified": theorem_gate,
        "principle_R_physical_zero_set_binding_certified": False,
        "law_III_critical_selection_derived": False,
        "all_scientific_gates_pass": False,
    }


def build_report(protocol_path: Path) -> dict[str, Any]:
    started = time.time()
    protocol = load_frozen_protocol(protocol_path)
    G = derive_tesc_hessian(protocol)
    theorem = audit_kernel_image_nullflow(G, protocol)
    return {
        "title": "K=1 downstream kernel-image null-flow theorem audit",
        "version": "3.8",
        "scientific_status": "CURRENT_DOWNSTREAM_K1_AUDIT_PHYSICAL_BINDING_OPEN",
        "protocol_sha256": protocol_sha256(protocol),
        "derivation_version": protocol.get("derivation_version"),
        "G_source": "protocol-derived frozen signed-TESC representative",
        "G": G,
        "native_two_line_structure": "N(G)=ker(B_c) union Im(B_c), B_c=J G-d_c I",
        "theorem": theorem,
        "interpretation": (
            "The audit certifies only the downstream K=1 auxiliary two-line "
            "algebraic theorem for the supplied Lorentzian G. It does not certify "
            "that the Principle-R realization-cost zero set equals these two lines "
            "and does not derive Law-III critical selection."
        ),
        "next_required_step": (
            "Independently define a Principle-R cost F_Pi and test whether "
            "Z(F_Pi)=ker(B_c) union Im(B_c) without constructing F_Pi from G, "
            "ker(B_c), or Im(B_c)."
        ),
        "claim_boundary": (
            "No Law-II/III physical derivation, spacetime, wavefunction, Born rule, "
            "collapse, hardware, or Principle-R physical zero-set binding claim."
        ),
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="protocols/frozen_tesc_protocol.json",
        help="Frozen R-to-Law-I protocol used only to derive the candidate G.",
    )
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    report = build_report(Path(args.protocol))
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "run_summary.json").write_text(
            json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
