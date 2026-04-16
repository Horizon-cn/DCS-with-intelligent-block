"""Lightweight helpers for task-space interpolation and waypoint path building.

The helpers in this module are intentionally dependency-free so they can be used
from simple scripts, future debug tools, or higher-level trajectory builders.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias

TaskPoint: TypeAlias = Sequence[float]
Waypoint: TypeAlias = Sequence[float | int | None]
EasingFn: TypeAlias = Callable[[float], float]


def smoothstep(t: float) -> float:
    """Return a smooth cubic easing value for ``t`` in the ``[0, 1]`` range."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def linear(t: float) -> float:
    """Return ``t`` unchanged after clamping to the ``[0, 1]`` range."""
    return max(0.0, min(1.0, float(t)))


def _resolve_easing(easing: str | EasingFn) -> EasingFn:
    if callable(easing):
        return easing
    if easing == "linear":
        return linear
    if easing == "smoothstep":
        return smoothstep
    raise ValueError(f"Unknown easing '{easing}'. Use 'linear', 'smoothstep', or a callable.")


def interpolate_task_space(
    start: TaskPoint,
    end: TaskPoint,
    samples: int = 2,
    *,
    easing: str | EasingFn = "smoothstep",
) -> list[list[float]]:
    """Interpolate smoothly between two task-space points.

    Args:
        start: Starting task-space point.
        end: Ending task-space point.
        samples: Number of points to return, including both endpoints.
        easing: Either ``"linear"``, ``"smoothstep"``, or a custom callable.
    """
    start_vec = [float(value) for value in start]
    end_vec = [float(value) for value in end]
    if len(start_vec) != len(end_vec):
        raise ValueError("start and end must have the same dimensionality")
    if len(start_vec) == 0:
        return []

    if samples <= 1:
        return [start_vec[:]]

    ease = _resolve_easing(easing)
    result: list[list[float]] = []
    for index in range(samples):
        alpha = index / (samples - 1)
        blend = ease(alpha)
        result.append(
            [s + (e - s) * blend for s, e in zip(start_vec, end_vec)]
        )
    return result


def interpolate_task_space_path(
    waypoints: Sequence[TaskPoint],
    samples_per_segment: int = 2,
    *,
    easing: str | EasingFn = "smoothstep",
) -> list[list[float]]:
    """Interpolate a smooth task-space path through a list of waypoints.

    The first waypoint is included once, and each following segment omits its
    first point so the path does not duplicate boundary samples.
    """
    if not waypoints:
        return []

    samples_per_segment = max(2, int(samples_per_segment))
    path: list[list[float]] = [list(map(float, waypoints[0]))]
    for start, end in zip(waypoints, waypoints[1:]):
        segment = interpolate_task_space(start, end, samples=samples_per_segment, easing=easing)
        path.extend(segment[1:])
    return path


def normalize_waypoint(waypoint: Waypoint) -> list[float | int | None]:
    """Normalize a waypoint to ``[j1, j2, j3, j4, j5, pwm1?, pwm2?]`` format."""
    values = list(waypoint)
    if len(values) < 5:
        raise ValueError("Waypoint must contain at least 5 joint values")
    if len(values) > 7:
        raise ValueError("Waypoint must contain at most 7 values")
    if len(values) == 5:
        values.extend([None, None])
    elif len(values) == 6:
        values.append(None)
    return values


def build_path(
    waypoints: Iterable[Waypoint],
    *,
    dedupe_consecutive: bool = True,
) -> list[list[float | int | None]]:
    """Normalize a sequence of waypoints into the interface file format."""
    path: list[list[float | int | None]] = []
    previous: list[float | int | None] | None = None

    for waypoint in waypoints:
        normalized = normalize_waypoint(waypoint)
        if dedupe_consecutive and previous == normalized:
            continue
        path.append(normalized)
        previous = normalized

    return path


def concat_paths(
    *paths: Iterable[Waypoint],
    dedupe_boundary: bool = True,
) -> list[list[float | int | None]]:
    """Concatenate multiple waypoint paths into one normalized path."""
    merged: list[list[float | int | None]] = []
    for path in paths:
        for waypoint in path:
            normalized = normalize_waypoint(waypoint)
            if dedupe_boundary and merged and merged[-1] == normalized:
                continue
            merged.append(normalized)
    return merged


__all__ = [
    "TaskPoint",
    "Waypoint",
    "EasingFn",
    "smoothstep",
    "linear",
    "interpolate_task_space",
    "interpolate_task_space_path",
    "normalize_waypoint",
    "build_path",
    "concat_paths",
]
