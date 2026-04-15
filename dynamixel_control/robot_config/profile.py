"""Central place to define the measured geometry for the 5-DOF inchworm robot."""

from __future__ import annotations

from kinematics_backend.robot_definition import Inchworm5DOFRobot
from typing import Sequence


JOINT_OFFSETS: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
    (24.0, 0.0, 29.0),   # J1 -> J2
    (73.0, 0.0, 0.0),  # J2 -> J3
    (73.0, 0.0, 0.0),   # J3 -> J4
    (24.0, 0.0, -29.0),   # J4 -> J5
)
BASE_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.0)
TOOL_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.0)
HOME_ANGLES_DEG: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
JOINT_LIMITS_DEG: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-90.0, 90.0),  # J1
    (-135.0, 45.0),  # J2
    (0.0, 180.0),  # J3
    (-135.0, 45.0),  # J4
    (-90.0, 90.0),  # J5
)
BASE_FORWARD_AXIS_LOCAL: tuple[float, float, float] = (1.0, 0.0, 0.0)
TOOL_FORWARD_AXIS_LOCAL: tuple[float, float, float] = (-1.0, 0.0, 0.0)
BASE_POINTING_AXIS_LOCAL: tuple[float, float, float] = (0.0, 0.0, 1.0)
TOOL_POINTING_AXIS_LOCAL: tuple[float, float, float] = (0.0, 0.0, -1.0)

# Maps IK-space joint angles into measured hardware-space servo angles.
# These values were fit from two anchor poses:
# - IK target position (85, 0, 0)  -> hardware [180.0, 224.4, 87.6, 219.0, 180.0]
# - IK target position (170, 0, 0) -> hardware [180.0, 181.4, 4.0, 181.1, 180.0]
# J1/J5 are pinned to 180 degrees at the anchors and their small deviations were
# treated as measurement noise, so they keep a simple +180 degree offset.
IK_TO_HARDWARE_SCALES: tuple[float, float, float, float, float] = (
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
)
IK_TO_HARDWARE_OFFSETS_DEG: tuple[float, float, float, float, float] = (
    180.0,
    147.4,
    -62.3,
    151.0,
    180.0,
)


def build_robot(
    *,
    joint_offsets: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
    joint_limits_deg: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    tool_offset: tuple[float, float, float] | None = None,
    base_offset: tuple[float, float, float] | None = None,
    home_angles_deg: tuple[float, float, float, float, float] | None = None,
    base_forward_axis_local: tuple[float, float, float] | None = None,
    tool_forward_axis_local: tuple[float, float, float] | None = None,
    base_pointing_axis_local: tuple[float, float, float] | None = None,
    tool_pointing_axis_local: tuple[float, float, float] | None = None,
) -> Inchworm5DOFRobot:
    """
    Build the project robot model from measured geometry.

    Edit the module constants above to define the project-default robot.
    The optional keyword arguments only exist for scripted experiments.

    Geometry meanings:
    - `JOINT_OFFSETS[0]`: vector from J1 axis to J2 axis, expressed in the J1-local frame
    - `JOINT_OFFSETS[1]`: vector from J2 axis to J3 axis, expressed in the J2-local frame
    - `JOINT_OFFSETS[2]`: vector from J3 axis to J4 axis, expressed in the J3-local frame
    - `JOINT_OFFSETS[3]`: vector from J4 axis to J5 axis, expressed in the J4-local frame
    - `JOINT_LIMITS_DEG`: 5 `(min_deg, max_deg)` pairs for J1..J5
    - `BASE_OFFSET`: world position of the J1 frame origin
    - `TOOL_OFFSET`: vector from the J5 frame origin to the active tool/contact point
    - `BASE_FORWARD_AXIS_LOCAL`: which local direction counts as the base's facing direction
    - `TOOL_FORWARD_AXIS_LOCAL`: which local direction counts as the tool's facing direction
    - `BASE_POINTING_AXIS_LOCAL`: the base gripper/contact normal used for feasibility checks
    - `TOOL_POINTING_AXIS_LOCAL`: the tool gripper/contact normal used for feasibility checks
    """
    return Inchworm5DOFRobot.from_joint_offsets(
        joint_offsets=joint_offsets or JOINT_OFFSETS,
        joint_limits_deg=joint_limits_deg or JOINT_LIMITS_DEG,
        tool_offset=TOOL_OFFSET if tool_offset is None else tool_offset,
        base_offset=BASE_OFFSET if base_offset is None else base_offset,
        home_angles_deg=HOME_ANGLES_DEG if home_angles_deg is None else home_angles_deg,
        base_forward_axis_local=BASE_FORWARD_AXIS_LOCAL if base_forward_axis_local is None else base_forward_axis_local,
        tool_forward_axis_local=TOOL_FORWARD_AXIS_LOCAL if tool_forward_axis_local is None else tool_forward_axis_local,
        base_pointing_axis_local=BASE_POINTING_AXIS_LOCAL if base_pointing_axis_local is None else base_pointing_axis_local,
        tool_pointing_axis_local=TOOL_POINTING_AXIS_LOCAL if tool_pointing_axis_local is None else tool_pointing_axis_local,
    )


def ik_to_hardware_angles(angles_deg: Sequence[float]) -> list[float]:
    """Convert 5 IK-space joint angles into measured hardware servo angles."""
    if len(angles_deg) != 5:
        raise ValueError("Expected 5 IK-space joint angles")

    return [
        offset_deg + scale * float(angle_deg)
        for angle_deg, scale, offset_deg in zip(
            angles_deg,
            IK_TO_HARDWARE_SCALES,
            IK_TO_HARDWARE_OFFSETS_DEG,
        )
    ]
