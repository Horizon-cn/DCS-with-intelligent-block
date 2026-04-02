"""
Robot Joint Motion Demo
演示 new_robot.urdf 中三个关节的依次运动

机器人结构:
- Joint 1: 由三个正交旋转关节组成 (j1_x, j1_y, j1_z)
- Joint 2: 单轴旋转关节 (j2_single)
- Joint 3: 由三个正交旋转关节组成 (j3_x, j3_y, j3_z)

演示: 依次控制这三个关节进行旋转运动
"""

import time
import math
import pybullet as p
from pathlib import Path
from config_loader import load_config
from simulation_setup import connect_and_configure, configure_visualizer, create_plane
from control.motion import _set_collision_with_all
from control.move import move_joint, move_rob_dir, move_rob_to_cube_side_xplus
from factory import *
import numpy as np
import random


def print_joint_info(robot_id: int) -> None:
    """打印机器人的关节信息"""
    num_joints = p.getNumJoints(robot_id)
    print(f"\n机器人总关节数: {num_joints}")
    print("-" * 80)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        print(f"关节 {i}: {joint_name} (类型: {joint_type})")
    print("-" * 80 + "\n")


def demo_joint_motion() -> None:
    """演示三个关节的顺序运动"""
    
    # 初始化模拟环境
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    
    connect_and_configure(sim_cfg)
    configure_visualizer(cfg["visualizer"], enable_rendering=True)
    
    # 创建平面
    plane_id = create_plane(cfg["plane"], sim_cfg["use_maximal_coordinates"])
    
    # 加载机器人
    new_robot_urdf = str(Path(__file__).resolve().parent.parent / "pybullet_data" / "new_robot.urdf")
    robot_id = p.loadURDF(
        new_robot_urdf,
        basePosition=[-3, -3, 1],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,  # 固定基座以便观察关节运动
    )

    cubev_shape_id, cubec_shape_id = create_cube_shapes(cfg["cube"])
    cube_id = create_cube(
        cubev_shape_id,
        cubec_shape_id,
        cfg["cube"],
        [-3, -3, 0.5],
        sim_cfg["use_maximal_coordinates"],
    )

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
    
    print("=" * 80)
    print("Robot Joint Motion Demo")
    print("=" * 80)
    
    # 打印关节信息
    print_joint_info(robot_id)
    
    # 等等配置完成
    for _ in range(100):
        p.stepSimulation()
        time.sleep(sim_cfg["time_step"])
    
    # 依次控制所有7个关节：-90° -> +180° -> -90°
    num_joints = p.getNumJoints(robot_id)
    # print(f"依次控制所有类型为0（revolute）的关节: -90° -> +180° -> -90°")
    # for joint_idx in range(num_joints):
    #     joint_info = p.getJointInfo(robot_id, joint_idx)
    #     joint_name = joint_info[1].decode('utf-8')
    #     joint_type = joint_info[2]
    #     if joint_type != 0:
    #         continue
    #     print(f"\n关节{joint_idx} [{joint_name}] 运动: -90° -> +180° -> -90° (类型: {joint_type})")
    #     cur_angle = p.getJointState(robot_id, joint_idx)[0]
    #     # 第一次：相对当前角度-90°
    #     move_joint(robot_id, joint_idx, cur_angle - math.pi/2, steps=200)
    #     print(f"   已完成 -90° 运动")
    #     time.sleep(1)
    #     # 第二次：相对当前角度+180°
    #     cur_angle = p.getJointState(robot_id, joint_idx)[0]
    #     move_joint(robot_id, joint_idx, cur_angle + math.pi, steps=400)
    #     print(f"   已完成 +180° 运动")
    #     time.sleep(1)
    #     # 第三次：相对当前角度-90°
    #     cur_angle = p.getJointState(robot_id, joint_idx)[0]
    #     move_joint(robot_id, joint_idx, cur_angle - math.pi/2, steps=200)
    #     print(f"   已完成 -90° 运动")
    #     time.sleep(1)
    # print("\n所有revolute关节演示完成！")

        # demo: 直接用索引控制关节
    # j3_y: 索引7, j2_single: 索引4, j1_y: 索引1
    # j3_y +30°

        # 1. 获取end_platform的link index（假设为9）



    # base_platform_idx = -1
    # end_platform_idx = 9

    # # 固定base_platform与地面plane
    # base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    # cid_base_plane = p.createConstraint(
    #     parentBodyUniqueId=robot_id,
    #     parentLinkIndex=base_platform_idx,
    #     childBodyUniqueId=plane_id,
    #     childLinkIndex=-1,
    #     jointType=p.JOINT_FIXED,
    #     jointAxis=[0,0,0],
    #     parentFramePosition=[0,0,0],
    #     childFramePosition=base_pos,
    #     parentFrameOrientation=base_orn,
    #     childFrameOrientation=[0,0,0,1]
    # )


    # cur = p.getJointState(robot_id, 0)[0]
    # move_joint(robot_id, 0, cur + math.radians(90), steps=120)

    # cur = p.getJointState(robot_id, 7)[0]
    # move_joint(robot_id, 7, cur + math.radians(30), steps=120)
    # print("j3_y(7) 已完成 +30° 运动")

    # # j2_single +120°
    # cur = p.getJointState(robot_id, 4)[0]
    # move_joint(robot_id, 4, cur + math.radians(120), steps=240)
    # print("j2_single(4) 已完成 +120° 运动")

    # # j1_x +30°
    # cur = p.getJointState(robot_id, 1)[0]
    # move_joint(robot_id, 1, cur + math.radians(30), steps=120)
    # print("j1_x(1) 已完成 +30° 运动")

    # # === 更换基座为end_platform并抬起base_platform使机器人竖直 ===


    # if end_platform_idx is not None:        
    #     # 先移除base_platform与地面的约束（如果有）
    #     p.removeConstraint(cid_base_plane)
    #     # 2. 获取end_platform当前世界位置
    #     link_state = p.getLinkState(robot_id, end_platform_idx)
    #     pos, orn = link_state[0], link_state[1]
    #     # 3. 创建约束，将end_platform与地面plane_id固定
    #     cid = p.createConstraint(
    #         parentBodyUniqueId=robot_id,
    #         parentLinkIndex=end_platform_idx,
    #         childBodyUniqueId=plane_id,
    #         childLinkIndex=-1,
    #         jointType=p.JOINT_FIXED,
    #         jointAxis=[0,0,0],
    #         parentFramePosition=[0,0,0],
    #         childFramePosition=pos,
    #         parentFrameOrientation=orn,
    #         childFrameOrientation=[0,0,0,1]
    #     )
    #     print(f"已将end_platform({end_platform_idx})与地面固定，约束id={cid}")

    #     # 切换基座后，依次让j3_y、j2_single、j1_y反向运动
    #     # j3_y: 索引7, j2_single: 索引4, j1_y: 索引1
    #     cur = p.getJointState(robot_id, 7)[0]
    #     move_joint(robot_id, 7, cur - math.radians(30), steps=120)
    #     print("j3_y(7) 已完成 -30° 运动")
    #     cur = p.getJointState(robot_id, 4)[0]
    #     move_joint(robot_id, 4, cur - math.radians(120), steps=240)
    #     print("j2_single(4) 已完成 -120° 运动")
    #     cur = p.getJointState(robot_id, 1)[0]
    #     move_joint(robot_id, 1, cur - math.radians(30), steps=120)
    #     print("j1_x(1) 已完成 -30° 运动")

    #     p.removeConstraint(cid)

    # # === 再次交换基座，重复一次相同方向的运动 ===
    # print("\n=== 再次交换基座，重复相同方向运动 ===")
    # # 1. 重新将base_platform与地面固定，解除end_platform与地面约束

    # # 获取end_platform当前世界位置
    # if end_platform_idx is not None:
    #     link_state = p.getLinkState(robot_id, end_platform_idx)
    #     pos, orn = link_state[0], link_state[1]
    #     # 重新固定end_platform与地面
    #     cid_end_plane2 = p.createConstraint(
    #         parentBodyUniqueId=robot_id,
    #         parentLinkIndex=end_platform_idx,
    #         childBodyUniqueId=plane_id,
    #         childLinkIndex=-1,
    #         jointType=p.JOINT_FIXED,
    #         jointAxis=[0,0,0],
    #         parentFramePosition=[0,0,0],
    #         childFramePosition=pos,
    #         parentFrameOrientation=orn,
    #         childFrameOrientation=[0,0,0,1]
    #     )

    # # 2. 依次让j3_y、j2_single、j1_x正向运动（与第一次方向一致）
    # cur = p.getJointState(robot_id, 1)[0]
    # move_joint(robot_id, 1, cur + math.radians(-30), steps=120)
    # print("j3_y(7) 已完成 +30° 运动 (base为base)")

    # cur = p.getJointState(robot_id, 4)[0]
    # move_joint(robot_id, 4, cur + math.radians(-120), steps=240)
    # print("j2_single(4) 已完成 +120° 运动 (base为base)")

    # cur = p.getJointState(robot_id, 7)[0]
    # move_joint(robot_id, 7, cur + math.radians(-30), steps=120)
    # print("j1_x(1) 已完成 +30° 运动 (base为base)")

    # # 3. 解除end_platform与地面的约束
    # p.removeConstraint(cid_end_plane2)

    # for _ in range(480):  # 假设仿真步长为1/240s，这里相当于1秒
    #     p.stepSimulation()
    #     time.sleep(sim_cfg["time_step"])

    # base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    # cid_base_plane = p.createConstraint(
    #     parentBodyUniqueId=robot_id,
    #     parentLinkIndex=base_platform_idx,
    #     childBodyUniqueId=plane_id,
    #     childLinkIndex=-1,
    #     jointType=p.JOINT_FIXED,
    #     jointAxis=[0,0,0],
    #     parentFramePosition=[0,0,0],
    #     childFramePosition=base_pos,
    #     parentFrameOrientation=base_orn,
    #     childFrameOrientation=[0,0,0,1]
    # )

    # cur = p.getJointState(robot_id, 1)[0]
    # move_joint(robot_id, 1, cur - math.radians(-30), steps=120)
    # print("j3_y(7) 已完成 -30° 运动")
    # cur = p.getJointState(robot_id, 4)[0]
    # move_joint(robot_id, 4, cur - math.radians(-120), steps=240)
    # print("j2_single(4) 已完成 -120° 运动")
    # cur = p.getJointState(robot_id, 7)[0]
    # move_joint(robot_id, 7, cur - math.radians(-30), steps=120)
    # print("j1_x(1) 已完成 -30° 运动")

    # p.removeConstraint(cid_base_plane)

    # for _ in range(480):  # 假设仿真步长为1/240s，这里相当于1秒
    #     p.stepSimulation()
    #     time.sleep(sim_cfg["time_step"])

    move_rob_to_cube_side_xplus(robot_id, plane_id, cube_id)

    # new_base = move_rob_dir(robot_id, -1, 30, plane_id)

    # for i in range(2):
    #     new_base = move_rob_dir(robot_id, new_base, 30, plane_id)
    # new_base = move_rob_dir(robot_id, new_base, 0, plane_id)

    for _ in range(200):
        p.stepSimulation()
        time.sleep(sim_cfg["time_step"])
    p.disconnect()


if __name__ == "__main__":
    demo_joint_motion()
