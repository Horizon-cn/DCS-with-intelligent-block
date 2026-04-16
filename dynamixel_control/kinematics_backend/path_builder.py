"""Build full joint-space trajectories from task-space waypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from trajectories.helpers import build_path, smoothstep

from .analytical_ik import AnalyticalIKSolver
from .ik_types import PoseTarget, PreloadOffset
from .robot_definition import Vector3


def _vec_lerp(start: Vector3, end: Vector3, alpha: float) -> Vector3:
    return tuple(start[index] + (end[index] - start[index]) * alpha for index in range(3))  # type: ignore[return-value]


def _vec_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_norm(vector: Vector3) -> float:
    return sum(component * component for component in vector) ** 0.5


def _vec_normalize(vector: Vector3 | None) -> Vector3 | None:
    if vector is None:
        return None
    norm = _vec_norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


@dataclass(frozen=True)
class TaskSpaceWaypoint:
    """A task-space point plus optional gripper and preload metadata."""

    position: Vector3
    forward_axis: Vector3 | None = None
    pwm1: float | int | None = None
    pwm2: float | int | None = None
    preload: PreloadOffset | None = None

    def with_world_preload(self, vector: Vector3) -> "TaskSpaceWaypoint":
        return TaskSpaceWaypoint(
            position=self.position,
            forward_axis=self.forward_axis,
            pwm1=self.pwm1,
            pwm2=self.pwm2,
            preload=PreloadOffset(vector=vector, frame="world"),
        )

    def with_tool_preload(self, local_vector: Vector3) -> "TaskSpaceWaypoint":
        return TaskSpaceWaypoint(
            position=self.position,
            forward_axis=self.forward_axis,
            pwm1=self.pwm1,
            pwm2=self.pwm2,
            preload=PreloadOffset(vector=local_vector, frame="tool"),
        )


class PathBuilder:
    """Convert task-space waypoint lists into OpenRB-style joint trajectories."""

    def __init__(self, solver: AnalyticalIKSolver) -> None:
        self.solver = solver

    def solve_waypoint(
        self,
        waypoint: TaskSpaceWaypoint,
        *,
        seed_angles_deg: Sequence[float] | None = None,
    ) -> list[float | int | None]:
        target_position = waypoint.position
        if waypoint.preload is not None:
            if waypoint.preload.frame == "world":
                target_position = _vec_add(target_position, waypoint.preload.vector)
            else:
                preload_world = self.solver.robot.rotate_local_vector(
                    seed_angles_deg or self.solver.robot.home_angles_deg,
                    waypoint.preload.vector,
                )
                target_position = _vec_add(target_position, preload_world)

        result = self.solver.solve(
            PoseTarget(position=target_position, forward_axis=waypoint.forward_axis),
            initial_angles_deg=seed_angles_deg,
        )
        if not result.success:
            raise ValueError(
                "IK did not converge for waypoint "
                f"{waypoint.position} (position error={result.position_error:.4f}, "
                f"orientation error={result.orientation_error_rad:.4f} rad)"
            )
        return [*result.angles_deg, waypoint.pwm1, waypoint.pwm2]

    def build_trajectory(
        self,
        waypoints: Sequence[TaskSpaceWaypoint],
        *,
        samples_per_segment: int = 2,
        easing=smoothstep,
        seed_angles_deg: Sequence[float] | None = None,
    ) -> list[list[float | int | None]]:
        if not waypoints:
            return []

        expanded = self.interpolate_waypoints(
            waypoints,
            samples_per_segment=samples_per_segment,
            easing=easing,
        )
        trajectory: list[list[float | int | None]] = []
        current_seed = list(seed_angles_deg or self.solver.robot.home_angles_deg)
        for waypoint in expanded:
            joint_waypoint = self.solve_waypoint(waypoint, seed_angles_deg=current_seed)
            trajectory.append(joint_waypoint)
            current_seed = [float(angle) for angle in joint_waypoint[:5]]
        return build_path(trajectory)

    def interpolate_waypoints(
        self,
        waypoints: Sequence[TaskSpaceWaypoint],
        *,
        samples_per_segment: int = 2,
        easing=smoothstep,
    ) -> list[TaskSpaceWaypoint]:
        if not waypoints:
            return []
        if len(waypoints) == 1:
            return [waypoints[0]]

        sample_count = max(2, int(samples_per_segment))
        expanded: list[TaskSpaceWaypoint] = [waypoints[0]]
        for start, end in zip(waypoints, waypoints[1:]):
            for index in range(1, sample_count):
                alpha = easing(index / (sample_count - 1))
                expanded.append(
                    TaskSpaceWaypoint(
                        position=_vec_lerp(start.position, end.position, alpha),
                        forward_axis=self._interp_axis(start.forward_axis, end.forward_axis, alpha),
                        pwm1=self._interp_pwm(start.pwm1, end.pwm1, alpha),
                        pwm2=self._interp_pwm(start.pwm2, end.pwm2, alpha),
                        preload=end.preload if index == sample_count - 1 else start.preload,
                    )
                )
        return expanded

    @staticmethod
    def _interp_axis(
        start: Vector3 | None,
        end: Vector3 | None,
        alpha: float,
    ) -> Vector3 | None:
        if start is None and end is None:
            return None
        if start is None:
            return _vec_normalize(end)
        if end is None:
            return _vec_normalize(start)
        return _vec_normalize(_vec_lerp(start, end, alpha))

    @staticmethod
    def _interp_pwm(
        start: float | int | None,
        end: float | int | None,
        alpha: float,
    ) -> float | int | None:
        if start is None and end is None:
            return None
        if start is None:
            return end
        if end is None:
            return start
        if alpha >= 1.0:
            return end
        return start


def build_task_space_trajectory(
    solver: AnalyticalIKSolver,
    waypoints: Iterable[TaskSpaceWaypoint],
    *,
    samples_per_segment: int = 2,
    seed_angles_deg: Sequence[float] | None = None,
) -> list[list[float | int | None]]:
    """Convenience wrapper for one-shot trajectory generation."""
    builder = PathBuilder(solver)
    return builder.build_trajectory(
        list(waypoints),
        samples_per_segment=samples_per_segment,
        seed_angles_deg=seed_angles_deg,
    )
