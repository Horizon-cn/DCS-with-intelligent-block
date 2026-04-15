"""Solve IK for a target pose and visualize the resulting chain."""

from __future__ import annotations

import argparse
import math

from kinematics_backend.analytical_ik import AnalyticalIKSolver
from kinematics_backend.ik_types import PoseTarget
from kinematics_backend.visualizer import plot_chain
from robot_config.profile import build_robot


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values")
    return tuple(parts)  # type: ignore[return-value]


def _rotation_matrix_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    ry = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rz = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    return _matmul(_matmul(rz, ry), rx)


def _matmul(a, b):
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(3):
        for col in range(3):
            out[row][col] = sum(a[row][k] * b[k][col] for k in range(3))
    return (
        (out[0][0], out[0][1], out[0][2]),
        (out[1][0], out[1][1], out[1][2]),
        (out[2][0], out[2][1], out[2][2]),
    )


def _matvec(matrix, vector):
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def _rpy_deg_from_rotation_matrix(rotation) -> tuple[float, float, float]:
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -rotation[2][0]))))
    if abs(rotation[2][0]) < 0.999999:
        roll = math.degrees(math.atan2(rotation[2][1], rotation[2][2]))
        yaw = math.degrees(math.atan2(rotation[1][0], rotation[0][0]))
    else:
        roll = 0.0
        yaw = math.degrees(math.atan2(-rotation[0][1], rotation[1][1]))
    return (roll, pitch, yaw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve IK for a target pose and visualize the result. "
            "Orientation is reduced to the desired tool facing direction because the robot has 5 DOF."
        ),
        epilog=(
            "PowerShell note: if a vector starts with a negative number, use '=' form, "
            "for example --forward-axis=-1,0,0"
        ),
    )
    parser.add_argument("--position", required=True, type=_parse_vec3, help="Target XYZ position, e.g. 100,0,20")
    parser.add_argument("--forward-axis", type=_parse_vec3, help="Desired tool-facing direction as a vector. In PowerShell, prefer --forward-axis=-1,0,0 for negative-leading values")
    parser.add_argument("--rpy-deg", type=_parse_vec3, help="Desired orientation as roll,pitch,yaw degrees")
    parser.add_argument("--joint-seed", type=lambda text: tuple(float(part.strip()) for part in text.split(",")), help="Optional 5-angle initial seed")
    parser.add_argument("--position-tol", type=float, default=2.5, help="Debug success tolerance for XYZ position error")
    parser.add_argument("--orientation-tol-deg", type=float, default=5.0, help="Debug success tolerance for tool-facing direction error in degrees")
    parser.add_argument("--text-only", action="store_true", help="Print the solved state without opening a plot")
    return parser


def _resolve_forward_axis(args, robot) -> tuple[float, float, float] | None:
    if args.forward_axis is not None:
        return args.forward_axis
    if args.rpy_deg is not None:
        rotation = _rotation_matrix_from_rpy_deg(*args.rpy_deg)
        return _matvec(rotation, robot.tool_forward_axis_local)
    return None


def main() -> None:
    args = _build_parser().parse_args()
    robot = build_robot()

    forward_axis = _resolve_forward_axis(args, robot)

    initial_angles = list(robot.home_angles_deg)
    if args.joint_seed is not None:
        initial_angles = list(args.joint_seed)
        if len(initial_angles) != 5:
            raise SystemExit("--joint-seed must contain exactly 5 comma-separated angles")
    target = PoseTarget(position=args.position, forward_axis=forward_axis)

    solver = AnalyticalIKSolver(
        robot,
        position_tolerance=args.position_tol,
        orientation_tolerance_rad=math.radians(args.orientation_tol_deg),
    )
    result = solver.solve(target, initial_angles_deg=initial_angles)

    final_position, final_rotation = robot.forward_kinematics(result.angles_deg)
    final_rpy_deg = _rpy_deg_from_rotation_matrix(final_rotation)

    print(f"success={result.success}")
    print(f"iterations={result.iterations}")
    print(f"angles_deg={[round(angle, 3) for angle in result.angles_deg]}")
    print(f"end_axis_separation_deg={result.end_axis_separation_deg:.3f}")
    print(
        "tool_pose_xyz_rpy_deg="
        f"[{final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f}, "
        f"{final_rpy_deg[0]:.3f}, {final_rpy_deg[1]:.3f}, {final_rpy_deg[2]:.3f}]"
    )

    if args.text_only:
        return

    title = "IK Debug View"
    if not result.success:
        title += " (did not fully converge)"
    plot_chain(
        robot,
        result.angles_deg,
        title=title,
        target_position=args.position,
        target_forward_axis=forward_axis,
    )


if __name__ == "__main__":
    main()
