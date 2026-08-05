"""Path-cost utilities for Principle-R zero-mode witnesses."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Array = np.ndarray
CostFunction = Callable[[Array, Array], float]
PathFunction = Callable[[float], Array]
VelocityFunction = Callable[[float], Array]


def evaluate_local_cost(cost: CostFunction, x: Array, velocity: Array) -> float:
    return float(cost(np.asarray(x, dtype=float), np.asarray(velocity, dtype=float)))


def _finite_difference_velocity(
    path: PathFunction,
    parameter: float,
    parameter_grid: Array,
) -> Array:
    grid = np.asarray(parameter_grid, dtype=float)
    index = int(np.argmin(np.abs(grid - parameter)))
    if index == 0:
        left, right = grid[0], grid[1]
    elif index == len(grid) - 1:
        left, right = grid[-2], grid[-1]
    else:
        left, right = grid[index - 1], grid[index + 1]
    return (np.asarray(path(float(right))) - np.asarray(path(float(left)))) / (right - left)


def accumulated_cost(
    cost: CostFunction,
    path: PathFunction,
    parameter_grid: Array,
    *,
    protocol: dict[str, Any],
    velocity: VelocityFunction | None = None,
) -> dict[str, Any]:
    grid = np.asarray(parameter_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("parameter_grid must contain at least two parameters")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("parameter_grid must be strictly increasing")
    if protocol["integration_method"] != "trapezoid":
        raise ValueError("only trapezoid integration is implemented")

    local_costs = []
    velocities = []
    for parameter in grid:
        x = np.asarray(path(float(parameter)), dtype=float)
        v = (
            np.asarray(velocity(float(parameter)), dtype=float)
            if velocity is not None
            else _finite_difference_velocity(path, float(parameter), grid)
        )
        velocities.append(v)
        local_costs.append(evaluate_local_cost(cost, x, v))
    costs = np.asarray(local_costs, dtype=float)
    total = float(np.trapezoid(costs, grid))
    minimum = float(np.min(costs))
    return {
        "parameter_grid": grid,
        "local_costs": costs,
        "velocities": np.asarray(velocities, dtype=float),
        "accumulated_cost": total,
        "finite_cost_values": bool(np.all(np.isfinite(costs))),
        "cost_nonnegative": bool(minimum >= -float(protocol["local_cost_zero_tol"])),
        "minimum_local_cost": minimum,
    }


def path_is_nonconstant(
    path: PathFunction,
    parameter_grid: Array,
    tolerance: float,
) -> bool:
    points = np.asarray([path(float(parameter)) for parameter in parameter_grid])
    return bool(np.max(np.linalg.norm(points - points[0], axis=1)) > tolerance)


def positive_measure_fraction(mask: Array, parameter_grid: Array) -> float:
    grid = np.asarray(parameter_grid, dtype=float)
    values = np.asarray(mask, dtype=bool)
    if len(values) != len(grid):
        raise ValueError("mask and parameter_grid lengths differ")
    if len(grid) < 2:
        return 0.0
    intervals = np.diff(grid)
    active = values[:-1] & values[1:]
    return float(np.sum(intervals[active]) / (grid[-1] - grid[0]))
