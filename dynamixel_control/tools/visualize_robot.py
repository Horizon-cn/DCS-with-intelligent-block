"""Minimal CLI for plotting the current robot geometry and joint offsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from kinematics_backend.visualizer import describe_chain, plot_chain
from robot_config.profile import build_robot


def _parse_angles(text: str) -> tuple[float, float, float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("Expected 5 comma-separated joint angles")
    return tuple(parts)  # type: ignore[return-value]


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values")
    return tuple(parts)  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the 5-DOF inchworm chain geometry.")
    parser.add_argument("--angles", type=_parse_angles, help="Optional joint-angle override for the current robot profile")
    parser.add_argument("--save", type=Path, help="Optional image output path")
    parser.add_argument("--text-only", action="store_true", help="Print chain points instead of opening a plot")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    robot = build_robot()
    angles = args.angles if args.angles is not None else tuple(robot.home_angles_deg)

    if args.text_only:
        print(describe_chain(robot, angles))
        return

    plot_chain(
        robot,
        angles,
        title="Inchworm Geometry Debug View",
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
