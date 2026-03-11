from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *
from config import scaling_factor


def to_grid(point):
        return tuple(int(round(v * scaling_factor)) for v in point)

def to_world(path):
        return [tuple(v / scaling_factor for v in p) for p in path]

def motionplan(sim, obj, map_, target_pos):

    start = to_grid(sim.getObjectPosition(obj, -1)[:2])
    goal = to_grid(target_pos[:2])
    print(f"Start: {start}, Goal: {goal}")
    
    planner = AStar(map_=map_, start=start, goal=goal)
    path, path_info = planner.plan()

    #vis = Visualizer2D()
    #vis.plot_grid_map(map_)
    #vis.plot_path(path, style="--", color="C4")
    #vis.show()
    #vis.close()

    world_path = to_world(path)
    #print(world_path)
    return world_path
