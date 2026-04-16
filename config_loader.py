from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "simulation": {
        "gravity": [0, 0, -10],
        "time_step": 1.0 / 240.0,
        "solver_iterations": 10,
        "use_maximal_coordinates": True,
        "settle_steps": 80,
        "post_move_steps": 500,
        "scaling_factor": 20.0
    },
    "plane": {
        "urdf": "plane100.urdf",
        "lateral_friction": 2.0,
    },
    "cube_org": {
        "mesh_file": "cube.obj",
        "visual_rgba": [1, 1, 1, 1],
        "visual_specular": [0.4, 0.4, 0],
        "frame_shift": [0, -0.02, 0],
        "cube_scale": [1, 1, 1],
        "mass": 1.0,
        "lateral_friction": 0.7,
    },
    "cube": {
        "mesh_file": "cube2.obj",
        "visual_rgba": [1, 1, 1, 1],
        "visual_specular": [0.4, 0.4, 0],
        "frame_shift": [0, 0, 0],
        "cube_scale": [0.70710, 0.7071, 0.7071],
        "mass": 1.0,
        "lateral_friction": 0.7,
    },
    "robot": {
        "mesh_file": "cube.obj",
        "visual_rgba": [0.8, 0.2, 0.2, 1],
        "visual_specular": [0.4, 0.4, 0],
        "frame_shift": [0, 0, 0],
        "robot_scale": [1, 1, 0.4],
        "mass": 5.0,
        "lateral_friction": 1.0,
    },
    "stack": {
        "count": 5,
        "base_position": [0, 0, 1],
        "z_spacing": 1,
    },
    "motion": {
        "x_offset": 1.5,
        "move_steps": 480,
    },
    "visualizer": {
        "disable_during_setup": {
            "rendering": True,
            "gui": True,
            "tiny_renderer": True,
        }
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    config = {
        section: value.copy() if isinstance(value, dict) else value
        for section, value in DEFAULT_CONFIG.items()
    }

    path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")
    if not path.exists():
        return config

    try:
        import yaml  # type: ignore
    except ImportError:
        # Fall back to defaults when PyYAML is not available.
        return config

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        return config

    return _deep_update(config, loaded)
