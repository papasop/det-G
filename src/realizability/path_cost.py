"""Path-cost utilities for Principle-R zero-mode witnesses."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Array = np.ndarray
CostFunction = Callable[[Array, Array], float]
PathFunction = Callable[[float], Array]
VelocityFunction = Callable[[float], Array]
SUPPORTED_DERIVATIVE_METHOD = (
    "second_order_finite_difference_from_path_points_with_velocity_crosscheck"
)


def evaluate_local_cost(cost: CostFunction, x: Array, velocity: Array) -> float:
    return float(cost(np.asarray(x, dtype=float), np.asarray(velocity, dtype=float)))


def second_order_path_derivative(parameter_grid: Array, path_points: Array) -> Array:
    grid = np.asarray(parameter_grid, dtype=float)
    points = np.asarray(path_points, dtype=float)
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError("parameter_grid must contain at least three parameters")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("parameter_grid must be strictly increasing")
    if points.ndim != 2 or len(points) != len(grid):
        raise ValueError("path_points must be a 2D array matching parameter_grid")
    return np.gradient(points, grid, axis=0, edge_order=2)


def accumulated_cost(
    cost: CostFunction,
    path: PathFunction,
    parameter_grid: Array,
    *,
    protocol: dict[str, Any],
    velocity: VelocityFunction | None = None,
) -> dict[str, Any]:
    grid = np.asarray(parameter_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError("parameter_grid must contain at least three parameters")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("parameter_grid must be strictly increasing")
    if protocol["integration_method"] != "trapezoid":
        raise ValueError("only trapezoid integration is implemented")
    if protocol["derivative_method"] != SUPPORTED_DERIVATIVE_METHOD:
        raise ValueError("only second-order path differentiation is implemented")

    local_costs = []
    path_points = np.asarray([path(float(parameter)) for parameter in grid], dtype=float)
    velocity_values = (
        np.asarray([velocity(float(parameter)) for parameter in grid], dtype=float)
        if velocity is not None
        else second_order_path_derivative(grid, path_points)
    )
    if velocity_values.shape != path_points.shape:
        raise ValueError("velocities must have the same shape as path points")
    for x, v in zip(path_points, velocity_values):
        local_costs.append(evaluate_local_cost(cost, x, v))
    costs = np.asarray(local_costs, dtype=float)
    total = float(np.trapezoid(costs, grid))
    minimum = float(np.min(costs))
    return {
        "parameter_grid": grid,
        "local_costs": costs,
        "velocities": velocity_values,
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
