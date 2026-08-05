"""Finite nonlinear zero-branch tracing for the frozen TESC cost."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .tesc import derive_tesc_hessian, signed_cost


def _bisect(function, left: float, right: float, tolerance: float = 1e-13) -> float:
    left_value = function(left)
    for _ in range(100):
        midpoint = (left + right) / 2
        midpoint_value = function(midpoint)
        if abs(midpoint_value) < tolerance or right - left < tolerance:
            return float(midpoint)
        if left_value * midpoint_value <= 0:
            right = midpoint
        else:
            left = midpoint
            left_value = midpoint_value
    return float((left + right) / 2)


def _roots_at_x(function, x_value: float, y_max: float, scan_step: float) -> list[float]:
    point_count = max(1001, int(math.ceil(2 * y_max / scan_step)) + 1)
    ys = np.linspace(-y_max, y_max, point_count)
    values = np.array([function(np.array([x_value, y])) for y in ys])
    roots: list[float] = []
    for index in range(point_count - 1):
        if abs(values[index]) < 1e-11:
            roots.append(float(ys[index]))
        if values[index] * values[index + 1] < 0:
            roots.append(_bisect(lambda y: function(np.array([x_value, y])), float(ys[index]), float(ys[index + 1])))
    roots.sort()
    deduped: list[float] = []
    for root in roots:
        if not deduped or abs(root - deduped[-1]) > 2 * scan_step:
            deduped.append(root)
    return deduped


def trace_finite_zero_branches(protocol: dict[str, Any]) -> dict[str, Any]:
    zero_cfg = protocol["finite_zero_set"]
    x_max = float(zero_cfg["abs_x"])
    initial_y_max = float(zero_cfg["initial_abs_y"])
    maximum_y_max = float(zero_cfg["maximum_abs_y"])
    growth = float(zero_cfg["growth"])
    sections = int(zero_cfg["sections"])
    scan_step = float(zero_cfg["scan_step"])
    function = lambda z: signed_cost(z, protocol)
    G = derive_tesc_hessian(protocol)
    discriminant = G[0, 1] ** 2 - G[0, 0] * G[1, 1]
    slopes = sorted(
        [
            (-G[0, 1] - math.sqrt(discriminant)) / G[1, 1],
            (-G[0, 1] + math.sqrt(discriminant)) / G[1, 1],
        ]
    )
    records = []
    for x_value in np.linspace(-x_max, x_max, sections):
        if abs(x_value) < 1e-14:
            continue
        y_max = initial_y_max
        history = []
        while True:
            roots = _roots_at_x(function, float(x_value), y_max, scan_step)
            history.append({"ymax": y_max, "root_count": len(roots), "roots": roots})
            if len(roots) >= 2 or y_max >= maximum_y_max - 1e-15:
                break
            y_max = min(maximum_y_max, y_max * growth)
        global_roots = roots if abs(y_max - maximum_y_max) < 1e-14 else _roots_at_x(function, float(x_value), maximum_y_max, scan_step)
        records.append(
            {
                "x": float(x_value),
                "adaptive_ymax": y_max,
                "adaptive_roots": roots,
                "global_roots": global_roots,
                "history": history,
                "max_residual": max([abs(function(np.array([x_value, y]))) for y in global_roots], default=None),
            }
        )
    counts = [len(record["global_roots"]) for record in records]
    recovered = [record for record in records if len(record["history"][0]["roots"]) < 2 and len(record["global_roots"]) == 2]
    missing = [record for record in records if len(record["global_roots"]) < 2]
    extra = [record for record in records if len(record["global_roots"]) > 2]
    maximum_residual = max((record["max_residual"] for record in records if record["max_residual"] is not None), default=float("inf"))
    near_origin = sorted([record for record in records if len(record["global_roots"]) == 2], key=lambda record: abs(record["x"]))[:8]
    tangent_error = max(
        (
            min(abs(y / record["x"] - slope) for slope in slopes)
            for record in near_origin
            for y in record["global_roots"]
        ),
        default=float("inf"),
    )
    gates = {
        "Lorentzian_local_Hessian": bool(np.linalg.det(G) < 0),
        "all_sections_have_two_roots": not missing and all(count >= 2 for count in counts),
        "no_extra_zero_branches_to_maximum_boundary": not extra and all(count <= 2 for count in counts),
        "all_root_residuals_small": maximum_residual < 1e-9,
        "branches_tangent_to_Hessian_null_rays": tangent_error < 0.15,
        "adaptive_expansion_recovers_initial_missing_roots": len(recovered) > 0,
    }
    return {
        "records": records,
        "gates": gates,
        "gate": all(gates.values()),
        "metrics": {
            "sections": len(records),
            "initial_missing_sections": sum(len(record["history"][0]["roots"]) < 2 for record in records),
            "recovered_sections": len(recovered),
            "still_missing_sections": len(missing),
            "extra_branch_sections": len(extra),
            "root_count_min": min(counts),
            "root_count_max": max(counts),
            "maximum_y_used": max(record["adaptive_ymax"] for record in records),
            "maximum_root_residual": maximum_residual,
            "tangent_slope_error": tangent_error,
            "signed_zero_contrast_search_domain": {"abs_x": x_max, "abs_y": maximum_y_max},
        },
    }
