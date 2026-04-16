"""Shared IK data structures used by the kinematics backend."""

from __future__ import annotations

from dataclasses import dataclass

from .robot_definition import Vector3


@dataclass(frozen=True)
class PoseTarget:
    """A partial end-effector target for the 5-DOF chain."""

    position: Vector3
    forward_axis: Vector3 | None = None


@dataclass(frozen=True)
class IKResult:
    """Inverse-kinematics result bundle."""

    success: bool
    angles_deg: list[float]
    iterations: int
    position_error: float
    orientation_error_rad: float
    end_axis_separation_deg: float


@dataclass(frozen=True)
class PreloadOffset:
    """A preload offset that can be added in world or local tool coordinates."""

    vector: Vector3
    frame: str = "world"

    def __post_init__(self) -> None:
        if self.frame not in {"world", "tool"}:
            raise ValueError("Preload frame must be 'world' or 'tool'")
