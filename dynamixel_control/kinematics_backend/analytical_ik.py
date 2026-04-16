"""Analytical IK for the structured 5-DOF inchworm chain."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .ik_types import IKResult, PoseTarget
from .robot_definition import Inchworm5DOFRobot, Vector3


def _vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_norm(vector: Vector3) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _vec_normalize(vector: Vector3) -> Vector3:
    norm = _vec_norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _angle_between(a: Vector3, b: Vector3) -> float:
    a_hat = _vec_normalize(a)
    b_hat = _vec_normalize(b)
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(a_hat, b_hat))))
    return math.acos(cosine)


def _rotation_z_deg(angle_deg: float):
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _rotation_y_deg(angle_deg: float):
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _matvec(matrix, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _transpose(matrix):
    return tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))


def _wrap_deg(angle_deg: float) -> float:
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _delta_deg(a: float, b: float) -> float:
    return _wrap_deg(a - b)


def _planar_link_params(offset: Vector3) -> tuple[float, float]:
    x, _, z = offset
    length = math.hypot(x, z)
    intrinsic_angle_deg = math.degrees(math.atan2(-z, x))
    return length, intrinsic_angle_deg


@dataclass(frozen=True)
class AnalyticalIKSolver:
    """Closed-form solver for the inchworm geometry used in this project."""

    robot: Inchworm5DOFRobot
    position_tolerance: float = 2.5
    orientation_tolerance_rad: float = math.radians(5.0)
    max_end_axis_separation_deg: float = 90.0

    def solve(
        self,
        target: PoseTarget,
        *,
        initial_angles_deg: Sequence[float] | None = None,
    ) -> IKResult:
        seed = list(initial_angles_deg or self.robot.home_angles_deg)
        candidates = self._candidate_solutions(target)
        if not candidates:
            return IKResult(False, list(seed), 1, float("inf"), float("inf"), self._end_axis_separation_deg(seed))

        best_result: IKResult | None = None
        best_score = float("inf")
        for candidate in candidates:
            result = self._evaluate(candidate, target)
            if result.end_axis_separation_deg > self.max_end_axis_separation_deg + 1e-9:
                continue
            score = (
                result.position_error
                + 20.0 * result.orientation_error_rad
                + 0.05 * sum(abs(_delta_deg(angle, ref)) for angle, ref in zip(candidate, seed))
            )
            if score < best_score:
                best_score = score
                best_result = result

        if best_result is None:
            return IKResult(False, list(seed), 1, float("inf"), float("inf"), self._end_axis_separation_deg(seed))

        success = (
            best_result.position_error <= self.position_tolerance
            and best_result.orientation_error_rad <= self.orientation_tolerance_rad
        )
        return IKResult(
            success,
            best_result.angles_deg,
            1,
            best_result.position_error,
            best_result.orientation_error_rad,
            best_result.end_axis_separation_deg,
        )

    def _candidate_solutions(self, target: PoseTarget) -> list[list[float]]:
        q1_candidates = self._candidate_q1_deg(target)
        solutions: list[list[float]] = []
        for q1_deg in q1_candidates:
            forward_body = self._target_forward_in_body(target.forward_axis, q1_deg)
            theta_deg, q5_deg = self._solve_theta_and_q5(forward_body)
            if theta_deg is None or q5_deg is None:
                continue

            planar_target = self._target_position_in_body(target.position, q1_deg)
            reduced_target = self._reduce_planar_target(planar_target, theta_deg, q5_deg)
            if reduced_target is None:
                continue

            x2, z2 = reduced_target
            l1, alpha1 = _planar_link_params(self.robot.joints[1].translation_after)
            l2, alpha2 = _planar_link_params(self.robot.joints[2].translation_after)
            reach = math.hypot(x2, z2)
            max_reach = l1 + l2
            min_reach = abs(l1 - l2)
            if reach > max_reach + 1e-6:
                scale = (max_reach - 1e-9) / reach
                x2 *= scale
                z2 *= scale
                reach = math.hypot(x2, z2)
            elif reach < min_reach - 1e-6:
                if reach < 1e-9:
                    x2 = min_reach
                    z2 = 0.0
                else:
                    scale = (min_reach + 1e-9) / reach
                    x2 *= scale
                    z2 *= scale
                reach = math.hypot(x2, z2)

            cos_gamma = max(-1.0, min(1.0, (x2 * x2 + z2 * z2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)))
            gamma_deg = math.degrees(math.acos(cos_gamma))
            beta_deg = math.degrees(math.atan2(-z2, x2))

            for elbow_sign in (1.0, -1.0):
                beta2_deg = elbow_sign * gamma_deg
                k1 = l1 + l2 * math.cos(math.radians(beta2_deg))
                k2 = l2 * math.sin(math.radians(beta2_deg))
                beta1_deg = beta_deg - math.degrees(math.atan2(k2, k1))

                q2_deg = beta1_deg - alpha1
                q3_deg = beta2_deg - (alpha2 - alpha1)
                q4_deg = theta_deg - q2_deg - q3_deg
                candidate = [q1_deg, q2_deg, q3_deg, q4_deg, q5_deg]
                if self.robot.are_angles_within_limits(candidate):
                    solutions.append(candidate)

        deduped: list[list[float]] = []
        for solution in solutions:
            if not any(all(abs(_delta_deg(a, b)) < 1e-4 for a, b in zip(solution, seen)) for seen in deduped):
                deduped.append(solution)
        return deduped

    def _candidate_q1_deg(self, target: PoseTarget) -> list[float]:
        candidates = [self.robot.home_angles_deg[0]]
        x, y, _ = target.position
        if abs(x) > 1e-9 or abs(y) > 1e-9:
            candidates.append(math.degrees(math.atan2(y, x)))
        if target.forward_axis is not None:
            fx, fy, _ = _vec_normalize(target.forward_axis)
            if abs(fx) > 1e-9 or abs(fy) > 1e-9:
                candidates.append(math.degrees(math.atan2(fy, fx)))

        deduped: list[float] = []
        for angle in candidates:
            wrapped = _wrap_deg(angle)
            if not any(abs(_delta_deg(wrapped, seen)) < 1e-6 for seen in deduped):
                deduped.append(wrapped)
        return deduped

    def _target_position_in_body(self, position_world: Vector3, q1_deg: float) -> Vector3:
        relative = _vec_sub(position_world, self.robot.base_offset)
        return _matvec(_transpose(_rotation_z_deg(q1_deg)), relative)

    def _target_forward_in_body(self, forward_axis_world: Vector3 | None, q1_deg: float) -> Vector3 | None:
        if forward_axis_world is None:
            return None
        return _matvec(_transpose(_rotation_z_deg(q1_deg)), _vec_normalize(forward_axis_world))

    def _solve_theta_and_q5(self, forward_body: Vector3 | None) -> tuple[float | None, float | None]:
        if forward_body is None:
            return (0.0, self.robot.home_angles_deg[4])

        fx, fy, fz = _vec_normalize(forward_body)
        # With TOOL_FORWARD_AXIS_LOCAL = (-1, 0, 0), the tool-facing model is:
        # fx = -cos(theta) * cos(q5)
        # fy = -sin(q5)
        # fz =  sin(theta) * cos(q5)
        #
        # Solving q5 from fy keeps the steering branch inside [-90, 90], which
        # matches the current hardware limits and avoids the 180-degree alias.
        q5_deg = math.degrees(math.asin(max(-1.0, min(1.0, -fy))))
        cos_q5 = math.cos(math.radians(q5_deg))

        # When q5 is near +/-90 degrees, cos(q5) is near zero and theta is
        # underdetermined by the requested facing vector alone. In that case we
        # keep the body on the home-like branch instead of letting atan2 pick an
        # arbitrary 180-degree flip.
        if abs(cos_q5) < 1e-6:
            theta_deg = 0.0
        else:
            theta_deg = math.degrees(math.atan2(fz, -fx))
        return (_wrap_deg(theta_deg), _wrap_deg(q5_deg))

    def _reduce_planar_target(self, planar_target: Vector3, theta_deg: float, q5_deg: float) -> tuple[float, float] | None:
        reduced = _vec_sub(planar_target, self.robot.joints[0].translation_after)
        reduced = _vec_sub(reduced, _matvec(_rotation_y_deg(theta_deg), self.robot.joints[3].translation_after))
        reduced = _vec_sub(
            reduced,
            _matvec(_rotation_y_deg(theta_deg), _matvec(_rotation_z_deg(q5_deg), self.robot.tool_offset)),
        )
        if abs(reduced[1]) > 1e-4:
            return None
        return (reduced[0], reduced[2])

    def _evaluate(self, angles_deg: Sequence[float], target: PoseTarget) -> IKResult:
        final_position, _ = self.robot.forward_kinematics(angles_deg)
        position_error = _vec_norm(_vec_sub(target.position, final_position))
        if target.forward_axis is None:
            orientation_error_rad = 0.0
        else:
            orientation_error_rad = _angle_between(
                self.robot.end_effector_forward_axis(angles_deg),
                _vec_normalize(target.forward_axis),
            )
        return IKResult(
            False,
            list(angles_deg),
            1,
            position_error,
            orientation_error_rad,
            self._end_axis_separation_deg(angles_deg),
        )

    def _end_axis_separation_deg(self, angles_deg: Sequence[float]) -> float:
        axes = self.robot.joint_axis_directions_world(angles_deg)
        return math.degrees(_angle_between(axes[0], axes[4]))
