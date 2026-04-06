from __future__ import annotations
import math
import time

import pybullet as p

from config_loader import load_config

cfg = load_config()
PI = math.pi


def _normalize_angle(angle: float) -> float:
    return (angle + PI) % (2 * PI) - PI

def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _animate_base_pose(
    body_id: int,
    steps: int,
    time_step: float,
    pose_at,
    *,
    zero_velocity: bool = False,
) -> None:
    for i in range(max(1, steps)):
        pos, orn = pose_at(smoothstep((i + 1) / max(1, steps)))
        p.resetBasePositionAndOrientation(body_id, pos, orn)
        if zero_velocity:
            p.resetBaseVelocity(body_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(time_step)

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
    

    cur_angle = p.getJointState(robot_id, joint_index)[0]
    direction = 1 if target_angle > cur_angle else -1
    tol = 0.01  
    max_steps = steps * 3
    max_force = 1000
    num_joints = p.getNumJoints(robot_id)

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

        slow_zone = 0.3 
        min_speed = 0.01 * abs(speed)

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
    _animate_base_pose(
        obj,
        move_steps,
        time_step,
        lambda s: ([
            start_pos[0] + (target_pos[0] - start_pos[0]) * s,
            start_pos[1] + (target_pos[1] - start_pos[1]) * s,
            start_pos[2] + (target_pos[2] - start_pos[2]) * s,
        ], start_orn),
    )

def move_base_tar(obj_id: int, target_pos: list[float], orn, steps: int, time_step: float) -> None:
    start_pos, _ = p.getBasePositionAndOrientation(obj_id)
    _animate_base_pose(
        obj_id,
        steps,
        time_step,
        lambda s: ([
            start_pos[0] + (target_pos[0] - start_pos[0]) * s,
            start_pos[1] + (target_pos[1] - start_pos[1]) * s,
            start_pos[2] + (target_pos[2] - start_pos[2]) * s,
        ], orn),
    )


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


    plane_pos, plane_orn = p.getBasePositionAndOrientation(plane_id)

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

    for _ in range(120):  
        p.stepSimulation()
        time.sleep(1/240)

    if base_id == base_platform_idx:
        link_state = p.getLinkState(robot_id, end_platform_idx)
        base_pos_new, base_orn_new = link_state[0], link_state[1]
    else:
        base_pos_new, base_orn_new = p.getBasePositionAndOrientation(robot_id)

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


    p.removeConstraint(cid_end_plane2)

    for _ in range(120):
        p.stepSimulation()
        time.sleep(1/240)

    if base_id == base_platform_idx:
        return end_platform_idx
    else:
        return base_platform_idx


def move_rob_to_cube_side(robot_id: int, plane_id: int, cube_id: int, direction_xy: str) -> int:
    """
    从 demo 初始状态出发:
    1. 先将 base_platform 固定在当前顶部位置；
    2. 按 XY 平面方向弯折机器人，让 end_platform 贴到对应侧面；
    3. 将 end_platform 固定到该侧面；
    4. 在 end_platform 固定后，将关节展开回初始角度。

    Args:
        direction_xy: 仅支持 "+X", "-X", "+Y", "-Y"。

    返回最终 end_platform 与 plane 的约束 id。
    """
    direction_key = direction_xy.strip().upper()
    yaw_by_dir = {
        "+X": 0.0,
        "-X": math.pi,
        "+Y": math.pi / 2,
        "-Y": -math.pi / 2,
    }
    if direction_key not in yaw_by_dir:
        raise ValueError(f"Unsupported direction_xy={direction_xy!r}, expected one of +X/-X/+Y/-Y")

    yaw_offset = yaw_by_dir[direction_key]

    base_platform_idx = -1
    end_platform_idx = 9

    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    platform_id = base_platform_idx
    platform_id_new = end_platform_idx

    # demo 初始状态中，机器人站在边长 1 的立方体上表面中心。
    # cube_half_extent = 0.5
    # platform_half_thickness = 0.05
    # side_target_pos = [
    #     base_pos[0] + cube_half_extent + platform_half_thickness,
    #     base_pos[1],
    #     cube_half_extent,
    # ]
    # side_delta_orn = p.getQuaternionFromEuler([0, +math.pi / 2, 0])
    # _, side_target_orn = p.multiplyTransforms([0, 0, 0], base_orn, [0, 0, 0], side_delta_orn)

    # ik_solution = p.calculateInverseKinematics(
    #     robot_id,
    #     end_platform_idx,
    #     side_target_pos,
    #     side_target_orn,
    #     maxNumIterations=200,
    #     residualThreshold=1e-5,
    # )

    # joint_targets = {
    #     0: ik_solution[0],
    #     1: ik_solution[1],
    #     4: ik_solution[3],
    #     7: ik_solution[5],
    #     8: ik_solution[6],
    # }

    # theta0 = joint_targets[0] - p.getJointState(robot_id, 0)[0]
    # theta1 = joint_targets[1] - p.getJointState(robot_id, 1)[0]
    # theta2 = joint_targets[4] - p.getJointState(robot_id, 4)[0]
    # theta4 = joint_targets[7] - p.getJointState(robot_id, 7)[0]
    # theta5 = joint_targets[8] - p.getJointState(robot_id, 8)[0]

    # print(f"Calculated IK joint deltas: theta0={math.degrees(theta0):.2f}°, theta1={math.degrees(theta1):.2f}°, "
    #       f"theta2={math.degrees(theta2):.2f}°, theta4={math.degrees(theta4):.2f}°, theta5={math.degrees(theta5):.2f}°")

    plane_pos, plane_orn = p.getBasePositionAndOrientation(plane_id)

    inv_plane_pos, inv_plane_orn = p.invertTransform(plane_pos, plane_orn)
    child_pos, child_orn = p.multiplyTransforms(inv_plane_pos, inv_plane_orn, base_pos, base_orn)
    cid_base_plane = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=platform_id,
        childBodyUniqueId=plane_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=child_pos,
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=child_orn
    )

    # Use joint-0 yaw to steer the same bending routine to one of 4 XY directions.
    cur0 = p.getJointState(robot_id, 0)[0]
    move_joint(robot_id, 0, cur0 + yaw_offset, steps=180)

    cur1 = p.getJointState(robot_id, 1)[0]
    move_joint(robot_id, 1, cur1 + math.radians(90), steps=180)

    cur2 = p.getJointState(robot_id, 4)[0]
    move_joint(robot_id, 4, cur2 + math.radians(90), steps=240)

    cur3 = p.getJointState(robot_id, 7)[0]
    move_joint(robot_id, 7, cur3 + math.radians(90), steps=180)

    # cur4 = p.getJointState(robot_id, 8)[0]
    # move_joint(robot_id, 8, cur4, steps=120)

    for _ in range(120):
        p.stepSimulation()
        time.sleep(1 / 240)

    end_state = p.getLinkState(robot_id, end_platform_idx)
    end_pos, end_orn = end_state[0], end_state[1]
    plane_pos, plane_orn = p.getBasePositionAndOrientation(plane_id)
    inv_plane_pos, inv_plane_orn = p.invertTransform(plane_pos, plane_orn)
    child_pos2, child_orn2 = p.multiplyTransforms(inv_plane_pos, inv_plane_orn, end_pos, end_orn)
    cid_end_cube = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=platform_id_new,
        childBodyUniqueId=plane_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=child_pos2,
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=child_orn2
    )

    p.removeConstraint(cid_base_plane)

    cur3 = p.getJointState(robot_id, 7)[0]
    move_joint(robot_id, 7, cur3 - math.radians(90), steps=180)

    cur2 = p.getJointState(robot_id, 4)[0]
    move_joint(robot_id, 4, cur2 - math.radians(90), steps=240)

    cur1 = p.getJointState(robot_id, 1)[0]
    move_joint(robot_id, 1, cur1 - math.radians(90), steps=180)

    cur0 = p.getJointState(robot_id, 0)[0]
    move_joint(robot_id, 0, cur0 - yaw_offset, steps=180)

    # cur4 = p.getJointState(robot_id, 8)[0]
    # move_joint(robot_id, 8, cur4 - math.radians(90), steps=120)

    # cur = p.getJointState(robot_id, 0)[0]
    # move_joint(robot_id, 0, cur, steps=120)

    for _ in range(120):
        p.stepSimulation()
        time.sleep(1 / 240)

    return cid_end_cube

def rotate_to(base_id: int, target_angle: float) -> None:
    sim_cfg = cfg["simulation"]
    motion_cfg = cfg["motion"]
    rotate_steps = motion_cfg["move_steps"]
    rotate_time_step = sim_cfg["time_step"]

    start_pos, start_orn = p.getBasePositionAndOrientation(base_id)
    _, _, start_yaw = p.getEulerFromQuaternion(start_orn)
    delta_angle = _normalize_angle(target_angle - start_yaw)

    _animate_base_pose(
        base_id,
        rotate_steps,
        rotate_time_step,
        lambda s: (start_pos, p.getQuaternionFromEuler([0, 0, start_yaw + delta_angle * s])),
        zero_velocity=True,
    )
