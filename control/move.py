from __future__ import annotations
import time
import pybullet as p
from config_loader import load_config

cfg = load_config()

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

def goto(robot_id: int, target_pos: list[float], speed: float = 0.05, tolerance: float = 0.01):
    """
    每次调用向目标位置前进一步（稳定速度）
    
    Args:
        robot_id: 机器人刚体 id
        target_pos: 目标位置 [x, y, z]
        speed: 每步移动的最大距离（速度）
        tolerance: 到达目标的容差（距离阈值）
        
    Returns:
        tuple: (是否到达, 当前位置)
    """
    cur_pos = p.getBasePositionAndOrientation(robot_id)[0]
    direction = [target_pos[i] - cur_pos[i] for i in range(3)]
    distance = sum(d**2 for d in direction) ** 0.5
    
    if distance <= tolerance:
        return (True, cur_pos)
    
    step = [d / distance * min(speed, distance) for d in direction]
    new_pos = [cur_pos[i] + step[i] for i in range(3)]
    _, cur_orn = p.getBasePositionAndOrientation(robot_id)
    p.resetBasePositionAndOrientation(robot_id, new_pos, cur_orn)
    
    return (False, new_pos)

def move_dir(obj: int, robot_orn, offset: float, move_steps: int, time_step: float) -> None:
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

def move_tar(obj_id: int, target_pos: list[float], orn, steps: int, time_step: float) -> None:
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


def rotate_base(base_id: int, angle: float) -> None:
    sim_cfg = cfg["simulation"]
    motion_cfg = cfg["motion"]
    rotate_steps = motion_cfg["move_steps"]
    rotate_time_step = sim_cfg["time_step"]

    start_pos, start_orn = p.getBasePositionAndOrientation(base_id)

    for i in range(rotate_steps):
        t = (i + 1) / rotate_steps
        s = smoothstep(t)
        delta_orn = p.getQuaternionFromEuler([0, 0, -angle * s])
        _, orn = p.multiplyTransforms([0, 0, 0], start_orn, [0, 0, 0], delta_orn)
        p.resetBasePositionAndOrientation(base_id, start_pos, orn)
        p.stepSimulation()
        time.sleep(rotate_time_step)