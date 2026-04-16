"""Simple trajectory entry point for OpenRB-150 execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Sequence

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from robot_config.profile import ik_to_hardware_angles

Trajectory = list[list[float]]
TrajectorySource = Trajectory | Sequence[Sequence[float]] | str | Path
SourceFrame = Literal["auto", "hardware", "ik"]
DEFAULT_DXL_IDS: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)


def _is_waypoint_list(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    for waypoint in value:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 5:
            return False
    return True


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load trajectory module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_trajectory(value: Sequence[Sequence[float]]) -> Trajectory:
    return [list(waypoint) for waypoint in value]


def load_trajectory(source: TrajectorySource, variable: str | None = None) -> Trajectory:
    """Load a trajectory from memory, JSON, or a small Python file."""
    if isinstance(source, (list, tuple)):
        return _coerce_trajectory(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not _is_waypoint_list(data):
            raise ValueError(f"{path} does not contain a waypoint list")
        return _coerce_trajectory(data)

    if suffix == ".py":
        module = _load_module(path)
        if variable:
            if not hasattr(module, variable):
                raise AttributeError(f"{path} has no '{variable}' value")
            data = getattr(module, variable)
            if not _is_waypoint_list(data):
                raise ValueError(f"{variable} in {path} is not a waypoint list")
            return _coerce_trajectory(data)

        for candidate_name in ("trajectory", "TRAJECTORY", "path", "PATH", "waypoints", "WAYPOINTS"):
            if hasattr(module, candidate_name):
                data = getattr(module, candidate_name)
                if _is_waypoint_list(data):
                    return _coerce_trajectory(data)

        for value in module.__dict__.values():
            if _is_waypoint_list(value):
                return _coerce_trajectory(value)

        raise ValueError(f"No waypoint list found in {path}")

    raise ValueError(f"Unsupported trajectory file type: {path.suffix}")


def _resolve_source_frame(source: TrajectorySource, source_frame: SourceFrame) -> Literal["hardware", "ik"]:
    if source_frame != "auto":
        return source_frame

    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".json":
        return "ik"
    return "hardware"


def _convert_waypoint_from_ik_to_hardware(waypoint: Sequence[float]) -> list[float]:
    hardware_angles = ik_to_hardware_angles(waypoint[:5])
    return [*hardware_angles, *waypoint[5:]]


def _convert_trajectory_from_ik_to_hardware(trajectory: Sequence[Sequence[float]]) -> Trajectory:
    return [_convert_waypoint_from_ik_to_hardware(waypoint) for waypoint in trajectory]


def _initialize_robot(bot: OpenRB150, ids: Sequence[int] = DEFAULT_DXL_IDS) -> None:
    print("Resetting Dynamixels...")
    if not bot.reset_all_dynamixels(list(ids)):
        raise RuntimeError("Failed to reset one or more Dynamixels")

    print("Applying Dynamixel config...")
    if not bot.apply_config_all_dynamixels(list(ids)):
        raise RuntimeError("Failed to apply config to one or more Dynamixels")

    print("Pinging servos...")
    missing_ids: list[int] = []
    for sid in ids:
        if bot.ping(sid):
            print(f"  ID {sid}: OK")
        else:
            print(f"  ID {sid}: NOT FOUND")
            missing_ids.append(sid)

    if missing_ids:
        missing_str = ", ".join(str(sid) for sid in missing_ids)
        raise RuntimeError(f"Missing Dynamixel response from IDs: {missing_str}")

    print("Enabling torque...")
    torque_failures: list[int] = []
    for sid in ids:
        if not bot.torque(sid, True):
            torque_failures.append(sid)

    if torque_failures:
        failure_str = ", ".join(str(sid) for sid in torque_failures)
        raise RuntimeError(f"Failed to enable torque on IDs: {failure_str}")


def _disable_torque(bot: OpenRB150, ids: Sequence[int] = DEFAULT_DXL_IDS) -> None:
    print("Disabling torque...")
    torque_failures: list[int] = []
    for sid in ids:
        if not bot.torque(sid, False):
            torque_failures.append(sid)

    if torque_failures:
        failure_str = ", ".join(str(sid) for sid in torque_failures)
        print(f"[WARN] Failed to disable torque on IDs: {failure_str}")


def execute_trajectory(
    source: TrajectorySource,
    *,
    port: str = "COM8",
    delay: float = 0.5,
    readback: bool = False,
    debug: bool = False,
    variable: str | None = None,
    source_frame: SourceFrame = "auto",
    initialize: bool = True,
) -> Trajectory:
    """Load a trajectory and send it to the robot."""
    trajectory = load_trajectory(source, variable=variable)
    resolved_frame = _resolve_source_frame(source, source_frame)
    if resolved_frame == "ik":
        trajectory = _convert_trajectory_from_ik_to_hardware(trajectory)

    if debug:
        print(f"[INFO] Loaded {len(trajectory)} waypoints in {resolved_frame} space.")
        return trajectory

    from hardware_control.openRB150interface import OpenRB150

    bot = OpenRB150(port=port)
    try:
        if initialize:
            _initialize_robot(bot)
        bot.execute_trajectory(trajectory, delay=delay, readback=readback)
    finally:
        _disable_torque(bot)
        bot.close()
    return trajectory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a trajectory on the OpenRB-150.")
    parser.add_argument("source", help="Path to a .py or .json trajectory file")
    parser.add_argument("--port", default="COM8", help="Serial port for the OpenRB-150")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between waypoints")
    parser.add_argument("--readback", action="store_true", help="Print joint readback after each step")
    parser.add_argument("--debug", action="store_true", help="Load and convert only; do not send to hardware")
    parser.add_argument("--variable", help="Named trajectory variable to load from a .py file")
    parser.add_argument("--skip-init", action="store_true", help="Skip reset/config/ping/torque preflight before execution")
    parser.add_argument(
        "--source-frame",
        choices=("auto", "hardware", "ik"),
        default="auto",
        help="Interpretation of joint values in the source file. Default: .json=ik, .py=hardware.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    execute_trajectory(
        args.source,
        port=args.port,
        delay=args.delay,
        readback=args.readback,
        debug=args.debug,
        variable=args.variable,
        source_frame=args.source_frame,
        initialize=not args.skip_init,
    )


if __name__ == "__main__":
    main()
