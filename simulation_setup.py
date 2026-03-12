from __future__ import annotations

import pybullet as p
import pybullet_data


def connect_and_configure(sim_cfg: dict) -> None:
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=sim_cfg["solver_iterations"])
    g = sim_cfg["gravity"]
    p.setGravity(g[0], g[1], g[2])


def configure_visualizer(visualizer_cfg: dict, enable_rendering: bool) -> None:
    disable_cfg = visualizer_cfg.get("disable_during_setup", {})
    if not enable_rendering:
        if disable_cfg.get("rendering", True):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if disable_cfg.get("gui", True):
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if disable_cfg.get("tiny_renderer", True):
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    else:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def create_plane(plane_cfg: dict, use_maximal_coordinates: bool) -> int:
    plane_id = p.loadURDF(
        plane_cfg["urdf"],
        useMaximalCoordinates=use_maximal_coordinates,
    )
    p.changeDynamics(plane_id, -1, lateralFriction=plane_cfg["lateral_friction"])
    return plane_id
