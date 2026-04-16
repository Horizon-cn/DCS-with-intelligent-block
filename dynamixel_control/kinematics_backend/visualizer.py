"""Simple visualization helpers for debugging robot geometry and offsets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Sequence

from .robot_definition import Inchworm5DOFRobot, Vector3


@dataclass(frozen=True)
class ChainSnapshot:
    """Computed geometry for one joint configuration."""

    angles_deg: list[float]
    points: list[Vector3]
    joint_axis_points: list[Vector3]
    joint_axis_directions: list[Vector3]
    base_forward_axis: Vector3
    tool_forward_axis: Vector3


def compute_chain_snapshot(robot: Inchworm5DOFRobot, angles_deg: Sequence[float]) -> ChainSnapshot:
    """Return the chain points for plotting or inspection."""
    return ChainSnapshot(
        angles_deg=[float(angle) for angle in angles_deg],
        points=robot.chain_points(angles_deg),
        joint_axis_points=robot.chain_points(angles_deg)[:-1],
        joint_axis_directions=robot.joint_axis_directions_world(angles_deg),
        base_forward_axis=robot.base_forward_axis_world(),
        tool_forward_axis=robot.end_effector_forward_axis(angles_deg),
    )


def plot_chain(
    robot: Inchworm5DOFRobot,
    angles_deg: Sequence[float],
    *,
    title: str = "Inchworm Chain",
    show_joint_labels: bool = True,
    target_position: Vector3 | None = None,
    target_forward_axis: Vector3 | None = None,
    save_path: str | Path | None = None,
) -> ChainSnapshot:
    """
    Plot the chain in 3D using matplotlib.

    This is meant for debugging geometry, offsets, and joint conventions.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'python -m pip install matplotlib'."
        ) from exc

    snapshot = compute_chain_snapshot(robot, angles_deg)
    xs = [point[0] for point in snapshot.points]
    ys = [point[1] for point in snapshot.points]
    zs = [point[2] for point in snapshot.points]

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(xs, ys, zs, marker="o", linewidth=2)
    axis.scatter(xs[0], ys[0], zs[0], s=70, label="base")
    axis.scatter(xs[-1], ys[-1], zs[-1], s=70, label="tool")

    arrow_scale = _arrow_scale(snapshot.points)
    _draw_arrow(axis, snapshot.points[0], snapshot.base_forward_axis, arrow_scale, "tab:green", "base facing")
    _draw_arrow(axis, snapshot.points[-1], snapshot.tool_forward_axis, arrow_scale, "tab:red", "tool facing")
    for index, (joint_point, joint_axis) in enumerate(zip(snapshot.joint_axis_points, snapshot.joint_axis_directions), start=1):
        label = "joint axis" if index == 1 else None
        _draw_arrow(axis, joint_point, joint_axis, 0.75 * arrow_scale, "tab:blue", label)
        axis.text(
            joint_point[0] + joint_axis[0] * 0.85 * arrow_scale,
            joint_point[1] + joint_axis[1] * 0.85 * arrow_scale,
            joint_point[2] + joint_axis[2] * 0.85 * arrow_scale,
            f"+{robot.joints[index - 1].axis.upper()}{index}",
            color="tab:blue",
        )

    if target_position is not None:
        axis.scatter(
            target_position[0],
            target_position[1],
            target_position[2],
            s=90,
            marker="x",
            color="tab:orange",
            label="target position",
        )
        axis.text(target_position[0], target_position[1], target_position[2], "target", color="tab:orange")
    if target_position is not None and target_forward_axis is not None:
        _draw_arrow(axis, target_position, target_forward_axis, arrow_scale, "tab:orange", "target facing")

    if show_joint_labels:
        axis.text(xs[0], ys[0], zs[0], "base")
        for index, point in enumerate(snapshot.points[1:-1], start=1):
            axis.text(point[0], point[1], point[2], f"J{index}")
        axis.text(xs[-1], ys[-1], zs[-1], "tool")

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(title)
    axis.legend()
    _set_equal_axes(axis, snapshot.points)

    if save_path is not None:
        figure.savefig(Path(save_path), dpi=160, bbox_inches="tight")

    plt.show()
    return snapshot


def describe_chain(robot: Inchworm5DOFRobot, angles_deg: Sequence[float]) -> str:
    """Return a simple text summary of all chain points for offset debugging."""
    snapshot = compute_chain_snapshot(robot, angles_deg)
    lines = [f"angles_deg={snapshot.angles_deg}"]
    for index, point in enumerate(snapshot.points):
        if index == 0:
            label = "base"
        elif index == len(snapshot.points) - 1:
            label = "tool"
        else:
            label = f"joint_{index}"
        lines.append(f"{label}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
    for index, axis_direction in enumerate(snapshot.joint_axis_directions, start=1):
        lines.append(
            f"joint_{index}_axis_+{robot.joints[index - 1].axis}: "
            f"({axis_direction[0]:.3f}, {axis_direction[1]:.3f}, {axis_direction[2]:.3f})"
        )
    lines.append(
        "base_facing: "
        f"({snapshot.base_forward_axis[0]:.3f}, {snapshot.base_forward_axis[1]:.3f}, {snapshot.base_forward_axis[2]:.3f})"
    )
    lines.append(
        "tool_facing: "
        f"({snapshot.tool_forward_axis[0]:.3f}, {snapshot.tool_forward_axis[1]:.3f}, {snapshot.tool_forward_axis[2]:.3f})"
    )
    return "\n".join(lines)


def _draw_arrow(axis, origin: Vector3, direction: Vector3, length: float, color: str, label: str | None) -> None:
    direction = _normalize(direction)
    axis.quiver(
        origin[0],
        origin[1],
        origin[2],
        direction[0],
        direction[1],
        direction[2],
        length=length,
        normalize=True,
        color=color,
        label=label,
    )


def _normalize(vector: Vector3) -> Vector3:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _arrow_scale(points: Sequence[Vector3]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    span = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        1.0,
    )
    return 0.2 * span


def _set_equal_axes(axis, points: Sequence[Vector3]) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]

    x_mid = (max(xs) + min(xs)) / 2.0
    y_mid = (max(ys) + min(ys)) / 2.0
    z_mid = (max(zs) + min(zs)) / 2.0
    radius = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        1.0,
    ) / 2.0

    axis.set_xlim(x_mid - radius, x_mid + radius)
    axis.set_ylim(y_mid - radius, y_mid + radius)
    axis.set_zlim(z_mid - radius, z_mid + radius)
