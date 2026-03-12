import time
import pybullet as p

from factory import *
from config_loader import load_config
from motion import *
from simulation_setup import connect_and_configure, configure_visualizer, create_plane


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
    cubes = create_cube_stack(
        cubev_shape_id,
        cubec_shape_id,
        cfg["cube"],
        cfg["stack"],
        sim_cfg["use_maximal_coordinates"],
    )
    cube_1 = create_cube(
        cubev_shape_id,
        cubec_shape_id,
        cfg["cube"],
        [2, 2, 2],
        sim_cfg["use_maximal_coordinates"],
    )
    cubes.append(cube_1)

    robv_shape_id, robc_shape_id = create_robot_shapes(cfg["robot"])
    robot1_id = create_robot(
        robv_shape_id,
        robc_shape_id,
        cfg["robot"],
        [2,2,1],
        sim_cfg["use_maximal_coordinates"],
    )
    robots_info.append(rob_info(robot_id=robot1_id, has_load=True, load_cube_id=cube_1))
    rob_num+=1

    robot2_id = create_robot(
        robv_shape_id,
        robc_shape_id,
        cfg["robot"],
        [-2,-2,1],
        sim_cfg["use_maximal_coordinates"],
    )
    robots_info.append(rob_info(robot_id=robot2_id))
    rob_num+=1

    obj_dict["plane"] = plane_id
    obj_dict["cubes"] = cubes
    

    configure_visualizer(cfg["visualizer"], enable_rendering=True)
    step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])

    robots_info[0].glue_to_cube(cube_1)

    top_cube = get_top_cube(cubes)
    robots_info[1].pick_up(top_cube, obj_dict)

    step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])
    robots_info[0].drop(obj_dict)

    step_simulation(sim_cfg["post_move_steps"], sim_cfg["time_step"])
    
    
    p.disconnect()


if __name__ == "__main__":
    main()
