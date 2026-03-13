import time
import pybullet as p
import python_motion_planning as pmp

from factory import *
from config_loader import load_config
from control.motion import get_top_cube
from simulation_setup import connect_and_configure, configure_visualizer, create_plane

cube_picked: dict[int, bool] = {}

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
    cube_stacks = []
    for y in range(1, 6):
        cube_stacks.append(
            create_cube_stack(
                cubev_shape_id,
                cubec_shape_id,
                cfg["cube"],
                cfg["stack"],
                sim_cfg["use_maximal_coordinates"],
                [8, y, 0],
            )
        )

    # Shared state for all cubes: {cube_id: is_picked}
    for stack in cube_stacks:
        for cube_id in stack:
            cube_picked[cube_id] = False

    robv_shape_id, robc_shape_id = create_robot_shapes(cfg["robot"])
    robot1_id = create_robot(
        robv_shape_id,
        robc_shape_id,
        cfg["robot"],
        [9.5,1,1],
        sim_cfg["use_maximal_coordinates"],
    )
    robots_info.append(rob_info(robot_id=robot1_id, cube_picked=cube_picked))
    rob_num+=1

    # robot2_id = create_robot(
    #     robv_shape_id,
    #     robc_shape_id,
    #     cfg["robot"],
    #     [9.5,3,1],
    #     sim_cfg["use_maximal_coordinates"],
    # )
    # robots_info.append(rob_info(robot_id=robot2_id))
    # rob_num+=1

    obj_dict["plane"] = plane_id
    #obj_dict["cubes"] = cubes


    configure_visualizer(cfg["visualizer"], enable_rendering=True)
    step_simulation(600, sim_cfg["time_step"])

    for i in range(len(cube_stacks)):
        j = 0
        while get_top_cube(cube_stacks[i]):
            top_cube = get_top_cube(cube_stacks[i])
            robots_info[0].pick_up(top_cube)

            step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])

            robots_info[0].rotate(3.14)
            robots_info[0].move_to([20-i, 20-j, 0], cube_stacks)
            robots_info[0].drop(obj_dict)
            robots_info[0].rotate(3.14)
            robots_info[0].move_to([9.5, 1+i, 0], cube_stacks)
            j+=1

    #top_cube1 = get_top_cube(cube_stacks[0])
    #top_cube2 = get_top_cube(cube_stacks[1])
    #robots_info[0].pick_up(top_cube1)
    #robots_info[1].pick_up(top_cube2)

    #step_simulation(sim_cfg["settle_steps"], sim_cfg["time_step"])
    #robots_info[0].rotate(3.14)
    #robots_info[0].drop(obj_dict)
    #robots_info[0].rotate(-3.14/2)
    #robots_info[0].move_to([20, 20, 0], cube_stacks)

    step_simulation(sim_cfg["post_move_steps"], sim_cfg["time_step"])
    
    
    p.disconnect()


if __name__ == "__main__":
    main()
