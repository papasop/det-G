#!/usr/bin/env python3
"""v3.1 algebraic audit: scalar obstruction and two-channel candidate.

This audit is intentionally separate from the public v0.1.0 entry point. It
does not scan numerical TESC parameters. It records the structural result that
a scalar Information-Time zero set supplies only one kernel line on a
two-dimensional plane, while a product of two independent channels supplies
two distinct zero lines and an indefinite quadratic representative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROTOCOL: dict[str, Any] = {
    "schema": "r-law1-two-channel-origin-audit-v3.1",
    "version": "0.2.0-preflight",
    "plane_dimension": 2,
    "scalar_information_time": {
        "formula": "F_IT(x,v)=abs(DPhi_x(v))/H(x)",
        "H_positive": True,
        "Dphi": [1.0, -0.5],
    },
    "two_channel_candidate": {
        "formula": "F_cross(x,v)=abs(L_plus(v) L_minus(v))/H(x)",
        "H_positive": True,
        "L_plus": [1.0, -1.0],
        "L_minus": [1.0, 1.0],
        "channels_defined_before_TESC_null_rays": False,
        "native_channel_provenance_supplied": False,
        "product_law_provenance_supplied": False,
    },
}


def as_vector(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (2,):
        raise ValueError("v3.1 only audits real two-dimensional channel covectors")
    return vector


def symmetric_product_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * (np.outer(left, right) + np.outer(right, left))


def audit(protocol: dict[str, Any]) -> dict[str, Any]:
    scalar = protocol["scalar_information_time"]
    candidate = protocol["two_channel_candidate"]

    dphi = as_vector(scalar["Dphi"])
    l_plus = as_vector(candidate["L_plus"])
    l_minus = as_vector(candidate["L_minus"])

    scalar_rank = int(np.linalg.matrix_rank(dphi.reshape(1, 2)))
    scalar_kernel_dimension = int(protocol["plane_dimension"] - scalar_rank)
    channel_matrix = np.vstack([l_plus, l_minus])
    channel_determinant = float(np.linalg.det(channel_matrix))
    channels_independent = bool(abs(channel_determinant) > 1e-12)

    G_cross = symmetric_product_matrix(l_plus, l_minus)
    detG = float(np.linalg.det(G_cross))
    eigenvalues = np.linalg.eigvalsh(G_cross)

    gates = {
        "scalar_Dphi_nonzero": bool(np.linalg.norm(dphi) > 1e-12),
        "scalar_zero_set_is_single_kernel_line": scalar_kernel_dimension == 1,
        "scalar_IT_cannot_supply_two_distinct_lines": scalar_kernel_dimension == 1,
        "two_channels_independent": channels_independent,
        "two_channel_zero_set_is_union_of_two_lines": channels_independent,
        "two_channel_product_detG_negative": detG < 0,
        "two_channel_product_signature_1_1": bool(eigenvalues[0] < 0 < eigenvalues[1]),
        "channels_defined_before_TESC_null_rays": bool(
            candidate["channels_defined_before_TESC_null_rays"]
        ),
        "native_channel_provenance_supplied": bool(candidate["native_channel_provenance_supplied"]),
        "product_law_provenance_supplied": bool(candidate["product_law_provenance_supplied"]),
    }

    return {
        "title": "R-Law I two-channel origin audit v3.1",
        "version": protocol["version"],
        "scientific_status": "SINGLE_CHANNEL_OBSTRUCTION_CERTIFIED_TWO_CHANNEL_MECHANISM_CONDITIONAL",
        "single_channel_obstruction": {
            "statement": (
                "A scalar Information-Time realization has zero set ker(DPhi_x), "
                "one linear subspace on a two-dimensional plane when DPhi_x is nonzero."
            ),
            "gate": gates["scalar_Dphi_nonzero"]
            and gates["scalar_zero_set_is_single_kernel_line"]
            and gates["scalar_IT_cannot_supply_two_distinct_lines"],
        },
        "two_channel_product_mechanism": {
            "statement": (
                "For independent L_plus and L_minus, abs(L_plus(v)L_minus(v))/H "
                "has zero set ker(L_plus) union ker(L_minus), and "
                "q(v)=L_plus(v)L_minus(v) has det(G)<0."
            ),
            "gate": gates["two_channels_independent"]
            and gates["two_channel_zero_set_is_union_of_two_lines"]
            and gates["two_channel_product_detG_negative"]
            and gates["two_channel_product_signature_1_1"],
        },
        "native_two_channel_origin": {
            "gate": gates["channels_defined_before_TESC_null_rays"]
            and gates["native_channel_provenance_supplied"]
            and gates["product_law_provenance_supplied"],
            "warning": "Defining L_plus and L_minus from already computed TESC null rays is circular.",
        },
        "gates": gates,
        "metrics": {
            "scalar_rank": scalar_rank,
            "scalar_kernel_dimension": scalar_kernel_dimension,
            "channel_determinant": channel_determinant,
            "G_cross": G_cross.tolist(),
            "detG_cross": detG,
            "eigenvalues_cross": eigenvalues.tolist(),
        },
        "claim_boundary": (
            "This audit certifies a scalar no-go and a conditional two-channel "
            "algebraic mechanism. It does not derive native L_plus, L_minus, "
            "their product law, or physical zero-set binding from Principle R."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v3.1 two-channel origin audit.")
    parser.add_argument("--outdir", default="reference_results/v0.2.0")
    args = parser.parse_args()

    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    result = audit(DEFAULT_PROTOCOL)

    (output / "protocol.json").write_text(json.dumps(DEFAULT_PROTOCOL, indent=2) + "\n")
    (output / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
