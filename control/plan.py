import pybullet as p
from python_motion_planning.controller import *
from python_motion_planning.common import *
from python_motion_planning.path_planner import AStar
from config_loader import load_config

cfg = load_config()
scaling_factor = cfg["simulation"]["scaling_factor"]

def motionplan(robot_id, map_, target_pos):

    start = tuple(int(round(v * scaling_factor)) for v in p.getBasePositionAndOrientation(robot_id)[0][:2])
    goal = tuple(int(round(v * scaling_factor)) for v in target_pos[:2])
    #print(f"Start: {start}, Goal: {goal}")
    
    planner = AStar(map_=map_, start=start, goal=goal)
    path, path_info = planner.plan()

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_path(path, style="--", color="C4")
    vis.show()
    vis.close()

    world_path = [tuple(v / scaling_factor for v in point) for point in path]
    #print(world_path)
    return world_path
