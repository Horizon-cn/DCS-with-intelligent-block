"""Generate an executable JSON trajectory from hard-coded task-space waypoints.

This script defines a small set of task-space keyframes plus gripper states,
interpolates smoothly between them in task space, solves IK for each sample,
and writes the resulting joint-space trajectory to JSON.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import TypeAlias

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from kinematics_backend import AnalyticalIKSolver, PathBuilder, TaskSpaceWaypoint
from robot_config.profile import build_robot


DEFAULT_OUTPUT = _PROJECT_DIR / "generated" / "task_space_trajectory.json"
DEFAULT_SAMPLES_PER_SEGMENT = 10
Number: TypeAlias = int | float
AxisVector: TypeAlias = tuple[Number, Number, Number]
AxisSpec: TypeAlias = str | AxisVector | list[Number] | None
KeyframeRow: TypeAlias = list[Number | AxisSpec]


def _parse_forward_axis(axis: AxisSpec) -> tuple[float, float, float] | None:
    """Return a 3D forward-axis vector from a tuple/list/string cell."""
    if isinstance(axis, str):
        axis = ast.literal_eval(axis)
    if axis is None:
        return None
    if len(axis) != 3:
        raise ValueError(f"Forward axis must have 3 values, got: {axis}")
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def _rows_to_keyframes(rows: list[KeyframeRow]) -> list[TaskSpaceWaypoint]:
    keyframes: list[TaskSpaceWaypoint] = []
    current_gripper1: Number | None = None
    current_gripper2: Number | None = None
    for row in rows:
        if len(row) == 6:
            x, y, z, forward_axis, current_gripper1, current_gripper2 = row
        elif len(row) == 4:
            x, y, z, forward_axis = row
            if current_gripper1 is None or current_gripper2 is None:
                raise ValueError(f"First row with omitted gripper values has nothing to inherit: {row}")
        else:
            raise ValueError(
                "Expected [x, y, z, forward_axis] or "
                f"[x, y, z, forward_axis, gripper1, gripper2], got: {row}"
            )
        keyframes.append(
            TaskSpaceWaypoint(
                position=(x, y, z),
                forward_axis=_parse_forward_axis(forward_axis),
                pwm1=current_gripper1,
                pwm2=current_gripper2,
            )
        )
    return keyframes


def build_keyframes() -> list[TaskSpaceWaypoint]:
    """Return task-space keyframes authored as compact editable rows."""
    rows: list[KeyframeRow] = [
        #    X,   Y,  Z, forward axis, gripper 1, gripper 2
        [85,   0,  0, (-1, 0, 0),       100,       100],
        [85,   0,  0, (-1, 0, 0),       180,       100],
        [85,   0, 35, (-1, 0, 0)],
        [170,  0, 35, (-1, 0, 0)],
        [170,  0,  0, (-1, 0, 0)],
        [170,  0,  0, (-1, 0, 0),       100,       100],
        [170,  0,  0, (-1, 0, 0),       180,       100],
        [170,  0, 35, (-1, 0, 0)],
        [85,   0, 35, (-1, 0, 0)],
        [85,  85, 35, (-1, 0, 0)],
        [0,   85, 35, (-1, 0, 0)],
        [85,  85, 35, (-1, 0, 0)],
        [85,   0, 35, (-1, 0, 0)],
        [85,   0,  0, (-1, 0, 0)],
        [85,   0,  0, (-1, 0, 0),       110,       110],
    ]
    return _rows_to_keyframes(rows)


def build_trajectory(samples_per_segment: int) -> list[list[float | int | None]]:
    robot = build_robot()
    solver = AnalyticalIKSolver(robot)
    builder = PathBuilder(solver)
    keyframes = build_keyframes()
    return builder.build_trajectory(
        keyframes,
        samples_per_segment=samples_per_segment,
    )


def write_trajectory_json(
    output_path: Path,
    trajectory: list[list[float | int | None]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a JSON trajectory from hard-coded task-space waypoints."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the generated JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--samples-per-segment",
        type=int,
        default=DEFAULT_SAMPLES_PER_SEGMENT,
        help=(
            "Number of task-space samples per segment, including the endpoint "
            f"(default: {DEFAULT_SAMPLES_PER_SEGMENT})"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory = build_trajectory(samples_per_segment=max(2, args.samples_per_segment))
    write_trajectory_json(args.output, trajectory)

    print(f"Wrote {len(trajectory)} waypoints to {args.output}")
    print("Execute it with:")
    print(f"  python -m trajectories.entry {args.output}")


if __name__ == "__main__":
    main()
