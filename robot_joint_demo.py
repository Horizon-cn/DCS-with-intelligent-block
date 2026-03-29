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


def move_joint(robot_id: int, joint_index: int, target_angle: float, 
               steps: int = 100, speed: float = 2.5) -> None:
    """
    平滑地移动指定关节到目标角度
    
    Args:
        robot_id: 机器人ID
        joint_index: 关节索引
        target_angle: 目标角度(弧度)
        steps: 运动步数
        speed: 关节运动速度
    """
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    time_step = sim_cfg["time_step"]
    
    # 先获取当前角度
    cur_angle = p.getJointState(robot_id, joint_index)[0]
    direction = 1 if target_angle > cur_angle else -1
    tol = 0.01  # 容许误差
    max_steps = steps * 3
    max_force = 500  # 增大力矩，防止抬不起来
    for i in range(max_steps):
        cur_angle = p.getJointState(robot_id, joint_index)[0]
        err = target_angle - cur_angle
        if abs(err) < tol:
            # 到达目标，停止
            p.setJointMotorControl2(
                robot_id,
                joint_index,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=max_force
            )
            break
        # 方向控制
        v = direction * abs(speed)
        # 如果接近目标，减速
        if abs(err) < 0.2:
            v *= 0.3
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            p.VELOCITY_CONTROL,
            targetVelocity=v,
            force=max_force
        )
        p.stepSimulation()
        time.sleep(time_step)
    # 最后确保速度为0
    p.setJointMotorControl2(
        robot_id,
        joint_index,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=max_force
    )


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
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True,  # 固定基座以便观察关节运动
    )
    
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
    print(f"依次控制所有类型为0（revolute）的关节: -90° -> +180° -> -90°")
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        if joint_type != 0:
            continue
        print(f"\n关节{joint_idx} [{joint_name}] 运动: -90° -> +180° -> -90° (类型: {joint_type})")
        cur_angle = p.getJointState(robot_id, joint_idx)[0]
        # 第一次：相对当前角度-90°
        move_joint(robot_id, joint_idx, cur_angle - math.pi/2, steps=200)
        print(f"   已完成 -90° 运动")
        time.sleep(1)
        # 第二次：相对当前角度+180°
        cur_angle = p.getJointState(robot_id, joint_idx)[0]
        move_joint(robot_id, joint_idx, cur_angle + math.pi, steps=400)
        print(f"   已完成 +180° 运动")
        time.sleep(1)
        # 第三次：相对当前角度-90°
        cur_angle = p.getJointState(robot_id, joint_idx)[0]
        move_joint(robot_id, joint_idx, cur_angle - math.pi/2, steps=200)
        print(f"   已完成 -90° 运动")
        time.sleep(1)
    print("\n所有revolute关节演示完成！")
    # 保持仿真运行，方便观察
    for _ in range(200):
        p.stepSimulation()
        time.sleep(sim_cfg["time_step"])
    p.disconnect()


if __name__ == "__main__":
    demo_joint_motion()
