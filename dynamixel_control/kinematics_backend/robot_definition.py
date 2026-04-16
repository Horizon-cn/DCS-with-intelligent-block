"""Robot-definition helpers for a simple 5-DOF inchworm serial chain."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence


Vector3 = tuple[float, float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _identity_matrix() -> Matrix3:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _matmul(a: Matrix3, b: Matrix3) -> Matrix3:
    out: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(3):
        for col in range(3):
            out[row][col] = sum(a[row][k] * b[k][col] for k in range(3))
    return (
        (out[0][0], out[0][1], out[0][2]),
        (out[1][0], out[1][1], out[1][2]),
        (out[2][0], out[2][1], out[2][2]),
    )


def _matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _vec_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(vector: Vector3, scale: float) -> Vector3:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _vec_norm(vector: Vector3) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _vec_normalize(vector: Vector3) -> Vector3:
    norm = _vec_norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return _vec_scale(vector, 1.0 / norm)


def _rotation_matrix(axis: str, angle_rad: float) -> Matrix3:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == "x":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    if axis == "z":
        return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    raise ValueError(f"Unsupported joint axis '{axis}'")


def _axis_unit_vector(axis: str) -> Vector3:
    if axis == "x":
        return (1.0, 0.0, 0.0)
    if axis == "y":
        return (0.0, 1.0, 0.0)
    if axis == "z":
        return (0.0, 0.0, 1.0)
    raise ValueError(f"Unsupported joint axis '{axis}'")


@dataclass(frozen=True)
class JointDefinition:
    """One revolute joint followed by a fixed translation to the next frame."""

    name: str
    axis: str
    translation_after: Vector3 = (0.0, 0.0, 0.0)
    min_angle_deg: float = -180.0
    max_angle_deg: float = 360.0

    def __post_init__(self) -> None:
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("Joint axis must be one of 'x', 'y', or 'z'")


@dataclass(frozen=True)
class Inchworm5DOFRobot:
    """Minimal serial-chain definition for the 5-DOF inchworm."""

    joints: Sequence[JointDefinition]
    base_offset: Vector3 = (0.0, 0.0, 0.0)
    tool_offset: Vector3 = (0.0, 0.0, 0.0)
    home_angles_deg: Sequence[float] = field(default_factory=lambda: (180.0, 180.0, 180.0, 180.0, 180.0))
    base_forward_axis_local: Vector3 = (1.0, 0.0, 0.0)
    tool_forward_axis_local: Vector3 = (0.0, 0.0, 1.0)
    base_pointing_axis_local: Vector3 = (0.0, 0.0, 1.0)
    tool_pointing_axis_local: Vector3 = (0.0, 0.0, -1.0)

    def __post_init__(self) -> None:
        if len(self.joints) != 5:
            raise ValueError("This helper expects exactly 5 revolute joints")
        if len(self.home_angles_deg) != 5:
            raise ValueError("home_angles_deg must contain 5 angles")

    @classmethod
    def from_joint_offsets(
        cls,
        *,
        joint_offsets: Sequence[Vector3],
        joint_limits_deg: Sequence[tuple[float, float]] | None = None,
        base_offset: Vector3 = (0.0, 0.0, 0.0),
        tool_offset: Vector3 = (0.0, 0.0, 0.0),
        home_angles_deg: Sequence[float] = (180.0, 180.0, 180.0, 180.0, 180.0),
        base_forward_axis_local: Vector3 = (1.0, 0.0, 0.0),
        tool_forward_axis_local: Vector3 = (0.0, 0.0, 1.0),
        base_pointing_axis_local: Vector3 = (0.0, 0.0, 1.0),
        tool_pointing_axis_local: Vector3 = (0.0, 0.0, -1.0),
    ) -> "Inchworm5DOFRobot":
        """
        Build a common inchworm layout with steering joints on Z at each end.

        Joint order:
        - J1: front gripper steering about Z
        - J2/J3/J4: body joints about X
        - J5: rear gripper steering about Z

        `joint_offsets` must contain 4 vectors:
        - J1 -> J2
        - J2 -> J3
        - J3 -> J4
        - J4 -> J5
        """
        if len(joint_offsets) != 4:
            raise ValueError("joint_offsets must contain exactly 4 vectors: J1->J2, J2->J3, J3->J4, J4->J5")
        if joint_limits_deg is None:
            joint_limits_deg = (
                (-180.0, 360.0),
                (-180.0, 360.0),
                (-180.0, 360.0),
                (-180.0, 360.0),
                (-180.0, 360.0),
            )
        if len(joint_limits_deg) != 5:
            raise ValueError("joint_limits_deg must contain exactly 5 (min_deg, max_deg) pairs")

        joints = (
            JointDefinition("front_steer", "z", joint_offsets[0], joint_limits_deg[0][0], joint_limits_deg[0][1]),
            JointDefinition("body_1", "y", joint_offsets[1], joint_limits_deg[1][0], joint_limits_deg[1][1]),
            JointDefinition("body_2", "y", joint_offsets[2], joint_limits_deg[2][0], joint_limits_deg[2][1]),
            JointDefinition("body_3", "y", joint_offsets[3], joint_limits_deg[3][0], joint_limits_deg[3][1]),
            JointDefinition("rear_steer", "z", (0.0, 0.0, 0.0), joint_limits_deg[4][0], joint_limits_deg[4][1]),
        )
        return cls(
            joints=joints,
            base_offset=base_offset,
            tool_offset=tool_offset,
            home_angles_deg=home_angles_deg,
            base_forward_axis_local=base_forward_axis_local,
            tool_forward_axis_local=tool_forward_axis_local,
            base_pointing_axis_local=base_pointing_axis_local,
            tool_pointing_axis_local=tool_pointing_axis_local,
        )

    def clamp_angles_deg(self, angles_deg: Sequence[float]) -> list[float]:
        return [
            max(joint.min_angle_deg, min(joint.max_angle_deg, angle))
            for joint, angle in zip(self.joints, angles_deg)
        ]

    def are_angles_within_limits(self, angles_deg: Sequence[float], *, atol: float = 1e-9) -> bool:
        if len(angles_deg) != len(self.joints):
            return False
        return all(
            joint.min_angle_deg - atol <= angle <= joint.max_angle_deg + atol
            for joint, angle in zip(self.joints, angles_deg)
        )

    def forward_kinematics(self, angles_deg: Sequence[float]) -> tuple[Vector3, Matrix3]:
        """Return end-effector position and orientation in the base frame."""
        if len(angles_deg) != 5:
            raise ValueError("Expected 5 joint angles")

        position = self.base_offset
        rotation = _identity_matrix()

        for joint, angle_deg in zip(self.joints, angles_deg):
            joint_rotation = _rotation_matrix(joint.axis, math.radians(angle_deg))
            rotation = _matmul(rotation, joint_rotation)
            position = _vec_add(position, _matvec(rotation, joint.translation_after))

        tool_position = _vec_add(position, _matvec(rotation, self.tool_offset))
        return tool_position, rotation

    def chain_points(self, angles_deg: Sequence[float]) -> list[Vector3]:
        """Return base, each joint/link endpoint, and the final tool point."""
        if len(angles_deg) != 5:
            raise ValueError("Expected 5 joint angles")

        points: list[Vector3] = [self.base_offset]
        position = self.base_offset
        rotation = _identity_matrix()

        for joint, angle_deg in zip(self.joints, angles_deg):
            joint_rotation = _rotation_matrix(joint.axis, math.radians(angle_deg))
            rotation = _matmul(rotation, joint_rotation)
            position = _vec_add(position, _matvec(rotation, joint.translation_after))
            points.append(position)

        points.append(_vec_add(position, _matvec(rotation, self.tool_offset)))
        return points

    def joint_axis_directions_world(self, angles_deg: Sequence[float]) -> list[Vector3]:
        """Return each joint's positive rotation axis in world coordinates."""
        if len(angles_deg) != 5:
            raise ValueError("Expected 5 joint angles")

        rotation = _identity_matrix()
        axes_world: list[Vector3] = []

        for joint, angle_deg in zip(self.joints, angles_deg):
            axes_world.append(_vec_normalize(_matvec(rotation, _axis_unit_vector(joint.axis))))
            joint_rotation = _rotation_matrix(joint.axis, math.radians(angle_deg))
            rotation = _matmul(rotation, joint_rotation)

        return axes_world

    def end_effector_forward_axis(self, angles_deg: Sequence[float]) -> Vector3:
        _, rotation = self.forward_kinematics(angles_deg)
        return _vec_normalize(_matvec(rotation, self.tool_forward_axis_local))

    def base_forward_axis_world(self) -> Vector3:
        """Return the configured base-frame forward direction in world coordinates."""
        return _vec_normalize(self.base_forward_axis_local)

    def base_pointing_axis_world(self) -> Vector3:
        """Return the configured base-frame pointing direction in world coordinates."""
        return _vec_normalize(self.base_pointing_axis_local)

    def end_effector_pointing_axis(self, angles_deg: Sequence[float]) -> Vector3:
        """Return the configured tool-frame pointing direction in world coordinates."""
        _, rotation = self.forward_kinematics(angles_deg)
        return _vec_normalize(_matvec(rotation, self.tool_pointing_axis_local))

    def rotate_local_vector(self, angles_deg: Sequence[float], local_vector: Vector3) -> Vector3:
        """Rotate a local tool-frame vector into the world frame."""
        _, rotation = self.forward_kinematics(angles_deg)
        return _matvec(rotation, local_vector)

    @staticmethod
    def as_degrees_list(angles_deg: Iterable[float]) -> list[float]:
        return [float(angle) for angle in angles_deg]
