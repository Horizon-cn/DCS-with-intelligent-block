import math
import random
import time
from pathlib import Path

import numpy as np
import pybullet as p

from config_loader import load_config
from control.dstar_surface_3d import DStarLiteSurface3D, FACES, NORM, Node
from control.motion import DynamicMoveToTargetTask
from factory import (
    create_cube_org,
    create_cube_shapes,
    create_cube_shapes_org,
    create_cube_stack,
    create_robot,
    create_robot_shapes,
    rob_info,
)
from simulation_setup import connect_and_configure, configure_visualizer, create_plane

cube_picked: dict[int, bool] = {}


def _quat_from_to(src: tuple[float, float, float], dst: tuple[float, float, float]):
    """Return quaternion (x, y, z, w) rotating unit vector src -> dst."""
    sx, sy, sz = src
    dx, dy, dz = dst
    dot = sx * dx + sy * dy + sz * dz

    # 180-degree case: pick a deterministic axis orthogonal to src.
    if dot < -0.999999:
        if abs(sx) < 0.9:
            ax, ay, az = 0.0, -sz, sy
        else:
            ax, ay, az = -sy, sx, 0.0
        n = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / n, ay / n, az / n
        return (ax, ay, az, 0.0)

    cx = sy * dz - sz * dy
    cy = sz * dx - sx * dz
    cz = sx * dy - sy * dx
    qw = 1.0 + dot
    n = math.sqrt(cx * cx + cy * cy + cz * cz + qw * qw)
    return (cx / n, cy / n, cz / n, qw / n)


def _spawn_pose_from_start_node(node: Node):
    """Map start node (position + face_dir) to robot base spawn pose.

    The robot starts exactly at the start-node face midpoint, and its base
    orientation is computed with ``-Z`` as the zero-rotation reference.
    """
    nx, ny, nz = NORM[node.face_dir]
    # Keep position identical to start.face_mid definition.
    pos = [node.pos[0] + 0.5 * (nx+1), node.pos[1] + 0.5 * (ny+1), node.pos[2] + 0.5 * nz]

    # Use -Z as the reference orientation: node.face_dir == -Z => identity quaternion.
    orn = _quat_from_to((0.0, 0.0, -1.0), (float(nx), float(ny), float(nz)))
    return pos, orn

def step_simulation(steps: int, time_step: float) -> None:
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(time_step)


def main() -> None:
    cfg = load_config()
    sim_cfg = cfg["simulation"]

    connect_and_configure(sim_cfg)
    configure_visualizer(cfg["visualizer"], enable_rendering=False)

    obj_dict = {}
    robots_info = []
    rob_num = 0

    plane_id = create_plane(cfg["plane"], sim_cfg["use_maximal_coordinates"])

    cubev_shape_id, cubec_shape_id = create_cube_shapes(cfg["cube"])
    # 2D stack grid: cube_stacks[x][y] -> one stack(list[int]).
    X, Y, Z = 5, 5, 5
    occ = np.zeros((X, Y, Z), dtype=np.uint8)
    cube_stacks = [[[] for _ in range(Y)] for _ in range(X)]
    for x in range(X):
        for y in range(Y):
            z = random.randint(1, Z-2)
            cube_stacks[x][y] = create_cube_stack(
                cubev_shape_id,
                cubec_shape_id,
                cfg["cube"],
                cfg["stack"],
                sim_cfg["use_maximal_coordinates"],
                z=z,
                base_pos=[(x + 1) * 1.0, (y + 1) * 1.0, 0.1],
            )
            occ[x, y, 0:z] = 1

    # Shared state for all cubes: {cube_id: is_picked}
    for x in range(len(cube_stacks)):
        for y in range(len(cube_stacks[x])):
            for cube_id in cube_stacks[x][y]:
                cube_picked[cube_id] = False

    valid_nodes = []
    for x in range(X):
        for y in range(Y):
            for zc in range(Z):
                if occ[x, y, zc] != 0:
                    continue
                for f in FACES:
                    ox, oy, oz = x + NORM[f][0], y + NORM[f][1], zc + NORM[f][2]
                    if 0 <= ox < X and 0 <= oy < Y and 0 <= oz < Z and occ[ox, oy, oz] == 1:
                        valid_nodes.append(Node((x + 0.5, y + 0.5, zc + 0.5), f))
    
    while True:
        start, goal = random.sample(valid_nodes, 2)
        if start.pos != goal.pos:
            break
    print(f"Randomly sampled start node: pos={start.pos}, face_dir={start.face_dir}")   

    cubev_shape_id, cubec_shape_id = create_cube_shapes_org(cfg["cube_org"])
    cube_id = create_cube_org(
        cubev_shape_id,
        cubec_shape_id,
        cfg["cube_org"],
        [9, 1, 1],
        sim_cfg["use_maximal_coordinates"],
    )

    new_robot_urdf = str(Path(__file__).resolve().parent.parent / "pybullet_data" / "new_robot.urdf")
    start_base_pos, start_base_orn = _spawn_pose_from_start_node(start)
    robot2_id = p.loadURDF(
        new_robot_urdf,
        basePosition=[1,8,1],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        globalScaling=1,
    )
    #_set_collision_with_all_links(robot2_id, enabled=False)
    
    robots_info.append(rob_info(robot_id=robot2_id, cube_picked=cube_picked))
    rob_num+=1

    robv_shape_id, robc_shape_id = create_robot_shapes(cfg["robot"])
    robot1_id = create_robot(
        robv_shape_id,
        robc_shape_id,
        cfg["robot"],
        [6.5,1,1],
        sim_cfg["use_maximal_coordinates"],
    )
    #robots_info.append(rob_info(robot_id=robot1_id, cube_picked=cube_picked))
    rob_num+=1

    robot_id = p.loadURDF(
        new_robot_urdf,
        basePosition=[8, 8, 0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
    )

    obj_dict["plane"] = plane_id
    #obj_dict["cubes"] = cubes

    configure_visualizer(cfg["visualizer"], enable_rendering=True)
    step_simulation(600, sim_cfg["time_step"])

    # new_base = move_rob_dir(robot_id, -1, 30, plane_id)

    # for i in range(2):
    #     new_base = move_rob_dir(robot_id, new_base, 30, plane_id)
    # new_base = move_rob_dir(robot_id, new_base, 0, plane_id)

    planner = DStarLiteSurface3D(occ, (X, Y, Z), start, goal)
    planner.plan_from_current()  # Initialize planning
    path = planner.extract_path_stateless(max_steps=100)
    planner.plot_3d_voxels_and_path(path)

    p.resetBasePositionAndOrientation(robot2_id, start_base_pos, start_base_orn)

    task = DynamicMoveToTargetTask(
        occ=occ,  # numpy array (X,Y,Z)
        size_xyz=(X, Y, Z),
        cube_stacks=cube_stacks,
        cube_picked=cube_picked,
        delta_per_step=0.002  # movement speed
    )

    task.replan_interval = 20  # Check map changes every 20 simulation steps

    task.setup(
        start_pos=start_base_pos,
        goal_pos=goal.pos,
        robot=robots_info[0],
        start_node=start,
        goal_node=goal,
    )

    task.begin()

    # for x in range(len(cube_stacks)):
    #     for y in range(len(cube_stacks[x])):
    #         stack = cube_stacks[x][y]
    #         j = 0
    #         while j < len(stack):
    #             top_cube = stack[-1 - j]
    #             robots_info[1].pick_up(top_cube)

    #             step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])

    #             robots_info[1].rotate(0)
    #             robots_info[1].move_to([10 + y * 1, 10 + j * 1, 0], cube_stacks)
    #             robots_info[1].rotate(math.pi)
    #             robots_info[1].drop(obj_dict)
    #             step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])
    #             robots_info[1].move_to([6.5 + x * 1, 1 + y * 1, 0], cube_stacks)
    #             j += 1

    #top_cube1 = get_top_cube(cube_stacks[0])
    #top_cube2 = get_top_cube(cube_stacks[1])
    #robots_info[0].pick_up(top_cube1)
    #robots_info[1].pick_up(top_cube2)

    #step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])
    #robots_info[0].rotate(math.pi)
    #robots_info[0].drop(obj_dict)
    #robots_info[0].rotate(-math.pi/2)
    #robots_info[0].move_to([20, 20, 0], cube_stacks)

    step_simulation(sim_cfg["post_move_steps"], sim_cfg["time_step"])
    
    
    p.disconnect()


if __name__ == "__main__":
    main()
