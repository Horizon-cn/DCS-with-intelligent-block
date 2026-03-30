from __future__ import annotations
import math
import time
import pybullet as p
from config_loader import load_config

cfg = load_config()


def _normalize_angle(angle: float) -> float:
    return (angle + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793

def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def step(sim, obj, axis_index, delta):
    """
    沿指定轴移动对象
    
    Args:
        sim: CoppeliaSim API 对象
        obj: 对象句柄
        axis_index: 轴索引 (0=x, 1=y, 2=z)
        delta: 移动增量
        
    Returns:
        list: 更新后的位置 [x, y, z]
    """
    p = sim.getObjectPosition(obj, -1)
    p[axis_index] += delta
    sim.setObjectPosition(obj, -1, p)
    return p

def goto(robot_id: int, target_pos: list[float], speed: float = 0.005, tolerance: float = 0.01):
    """
    每次调用为下一帧设置匀速速度，使机器人连续朝目标移动
    
    Args:
        robot_id: 机器人刚体 id
        target_pos: 目标位置 [x, y, z]
        speed: 每个仿真步允许前进的最大距离
        tolerance: 到达目标的容差（距离阈值）
        
    Returns:
        tuple: (是否到达, 当前位置)
    """
    time_step = 1/2400
    cur_pos = p.getBasePositionAndOrientation(robot_id)[0]
    direction = [target_pos[i] - cur_pos[i] for i in range(3)]
    distance = sum(d**2 for d in direction) ** 0.5
    
    if distance <= tolerance:
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        return (True, cur_pos)
    
    step_distance = min(speed, distance)
    linear_speed = step_distance / time_step
    velocity = [d / distance * linear_speed for d in direction]
    p.resetBaseVelocity(robot_id, linearVelocity=velocity, angularVelocity=[0, 0, 0])
    
    return (False, cur_pos)

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
    max_force = 1000
    num_joints = p.getNumJoints(robot_id)
    # 锁定除当前关节外的所有关节
    for j in range(num_joints):
        if j == joint_index:
            continue
        joint_info = p.getJointInfo(robot_id, j)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            cur_pos = p.getJointState(robot_id, j)[0]
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=cur_pos,
                force=max_force
            )
    for i in range(max_steps):
        cur_angle = p.getJointState(robot_id, joint_index)[0]
        err = target_angle - cur_angle
        dist = abs(err)
        # 设定减速区间
        slow_zone = 0.3  # 距目标小于此值开始减速
        min_speed = 0.01 * abs(speed)
        # 速度规划：距离越近速度越小，到达目标前速度已降为0
        if dist < tol:
            p.setJointMotorControl2(
                robot_id,
                joint_index,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=max_force
            )
            break
        if dist < slow_zone:
            v = direction * max(min_speed, abs(speed) * (dist / slow_zone))
        else:
            v = direction * abs(speed)
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

def move_base_dir(obj: int, robot_orn, offset: float, move_steps: int, time_step: float) -> None:
    start_pos, start_orn = p.getBasePositionAndOrientation(obj)

    # Forward is robot local +X axis; convert it into world coordinates.
    rot = p.getMatrixFromQuaternion(robot_orn)
    forward = [rot[0], rot[3], rot[6]]
    target_pos = [
        start_pos[0] + offset * forward[0],
        start_pos[1] + offset * forward[1],
        start_pos[2] + offset * forward[2],
    ]

    for i in range(move_steps):
        t = (i + 1) / move_steps
        s = smoothstep(t)
        pos = [
            start_pos[0] + (target_pos[0] - start_pos[0]) * s,
            start_pos[1] + (target_pos[1] - start_pos[1]) * s,
            start_pos[2] + (target_pos[2] - start_pos[2]) * s,
        ]
        p.resetBasePositionAndOrientation(obj, pos, start_orn)
        p.stepSimulation()
        time.sleep(time_step)

def move_base_tar(obj_id: int, target_pos: list[float], orn, steps: int, time_step: float) -> None:
    start_pos, _ = p.getBasePositionAndOrientation(obj_id)
    for i in range(steps):
        t = (i + 1) / steps
        s = smoothstep(t)
        pos = [
            start_pos[0] + (target_pos[0] - start_pos[0]) * s,
            start_pos[1] + (target_pos[1] - start_pos[1]) * s,
            start_pos[2] + (target_pos[2] - start_pos[2]) * s,
        ]
        p.resetBasePositionAndOrientation(obj_id, pos, orn)
        p.stepSimulation()
        time.sleep(time_step)


def move_rob_dir(robot_id: int, base_id: int, ang: float, plane_id: int) -> None:
    base_platform_idx = -1
    end_platform_idx = 9

    if base_id == base_platform_idx:
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        link_state = p.getLinkState(robot_id, end_platform_idx)
        platform_id = base_id
        platform_id_new = end_platform_idx

        theta0 = math.radians(ang)
        theta1 = math.radians(30)
        theta2 = math.radians(120)

        joint_ids = [0, 7, 4, 1]

    else:
        link_state = p.getLinkState(robot_id, end_platform_idx)
        base_pos, base_orn = link_state[0], link_state[1]

        platform_id = base_id
        platform_id_new = base_platform_idx

        theta0 = math.radians(ang)
        theta1 = -math.radians(30)
        theta2 = -math.radians(120)

        joint_ids = [8, 1, 4, 7]

    # --- 修正：用当前实际相对位姿 ---
    # 获取地面（plane_id）世界位姿
    plane_pos, plane_orn = p.getBasePositionAndOrientation(plane_id)
    # 计算plane在link下的相对位姿
    inv_link_pos, inv_link_orn = p.invertTransform(base_pos, base_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_link_pos, inv_link_orn, plane_pos, plane_orn)
    cid_base_plane = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=platform_id,
        childBodyUniqueId=plane_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0,0,0],
        parentFramePosition=rel_pos,
        childFramePosition=[0,0,0],
        parentFrameOrientation=rel_orn,
        childFrameOrientation=[0,0,0,1]
    )

    cur = p.getJointState(robot_id, joint_ids[0])[0]
    move_joint(robot_id, joint_ids[0], cur + theta0, steps=120)
    
    cur1 = p.getJointState(robot_id, joint_ids[1])[0]
    move_joint(robot_id, joint_ids[1], cur1 + theta1, steps=120)

    # j2_single +120°
    cur2 = p.getJointState(robot_id, joint_ids[2])[0]
    move_joint(robot_id, joint_ids[2], cur2 + theta2, steps=240)

    # j1_x +30°
    cur3 = p.getJointState(robot_id, joint_ids[3])[0]
    move_joint(robot_id, joint_ids[3], cur3 + theta1, steps=120)

    p.removeConstraint(cid_base_plane)

    for _ in range(120):  # 假设仿真步长为1/240s，这里相当于1秒
        p.stepSimulation()
        time.sleep(1/240)

    if base_id == base_platform_idx:
        link_state = p.getLinkState(robot_id, end_platform_idx)
        base_pos_new, base_orn_new = link_state[0], link_state[1]
    else:
        base_pos_new, base_orn_new = p.getBasePositionAndOrientation(robot_id)

    # --- 修正：用当前实际相对位姿 ---
    plane_pos2, plane_orn2 = p.getBasePositionAndOrientation(plane_id)
    inv_link_pos2, inv_link_orn2 = p.invertTransform(base_pos_new, base_orn_new)
    rel_pos2, rel_orn2 = p.multiplyTransforms(inv_link_pos2, inv_link_orn2, plane_pos2, plane_orn2)
    cid_end_plane2 = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=platform_id_new,
        childBodyUniqueId=plane_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0,0,0],
        parentFramePosition=rel_pos2,
        childFramePosition=[0,0,0],
        parentFrameOrientation=rel_orn2,
        childFrameOrientation=[0,0,0,1]
    )

    cur1 = p.getJointState(robot_id, joint_ids[1])[0]
    move_joint(robot_id, joint_ids[1], cur1 - theta1, steps=120)

    cur2 = p.getJointState(robot_id, joint_ids[2])[0]
    move_joint(robot_id, joint_ids[2], cur2 - theta2, steps=240)

    cur3 = p.getJointState(robot_id, joint_ids[3])[0]
    move_joint(robot_id, joint_ids[3], cur3 - theta1, steps=120)

    cur = p.getJointState(robot_id, joint_ids[0])[0]
    move_joint(robot_id, joint_ids[0], cur - theta0, steps=120)


    # 3. 解除end_platform与地面的约束
    p.removeConstraint(cid_end_plane2)

    for _ in range(120):  # 假设仿真步长为1/240s，这里相当于1秒
        p.stepSimulation()
        time.sleep(1/240)

    if base_id == base_platform_idx:
        return end_platform_idx
    else:
        return base_platform_idx



def rotate_base(base_id: int, angle: float) -> None:
    sim_cfg = cfg["simulation"]
    motion_cfg = cfg["motion"]
    rotate_steps = motion_cfg["move_steps"]
    rotate_time_step = sim_cfg["time_step"]

    start_pos, start_orn = p.getBasePositionAndOrientation(base_id)
    p.resetBaseVelocity(base_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    for i in range(rotate_steps):
        t = (i + 1) / rotate_steps
        s = smoothstep(t)
        delta_orn = p.getQuaternionFromEuler([0, 0, -angle * s])
        _, orn = p.multiplyTransforms([0, 0, 0], start_orn, [0, 0, 0], delta_orn)
        p.resetBasePositionAndOrientation(base_id, start_pos, orn)
        p.resetBaseVelocity(base_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(rotate_time_step)

def rotate_to(base_id: int, target_angle: float) -> None:
    sim_cfg = cfg["simulation"]
    motion_cfg = cfg["motion"]
    rotate_steps = motion_cfg["move_steps"]
    rotate_time_step = sim_cfg["time_step"]

    start_pos, start_orn = p.getBasePositionAndOrientation(base_id)
    roll, pitch, start_yaw = p.getEulerFromQuaternion(start_orn)
    delta_angle = _normalize_angle(target_angle - start_yaw)

    for i in range(rotate_steps):
        t = (i + 1) / rotate_steps
        s = smoothstep(t)
        current_angle = start_yaw + delta_angle * s
        current_orn = p.getQuaternionFromEuler([0, 0, current_angle])
        p.resetBasePositionAndOrientation(base_id, start_pos, current_orn)
        p.resetBaseVelocity(base_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(rotate_time_step)