from __future__ import annotations
import time
import pybullet as p
from config_loader import load_config

cfg = load_config()


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def get_top_cube(cubes: list[int]) -> int:
    return max(cubes, key=lambda cid: p.getBasePositionAndOrientation(cid)[0][2])


def set_cube_collisions(top_cube: int, base: int, plane_id: int, enabled: bool) -> None:
    flag = 1 if enabled else 0
    p.setCollisionFilterPair(top_cube, base, -1, -1, flag)
    p.setCollisionFilterPair(top_cube, plane_id, -1, -1, flag)

def _set_collision_with_all(body_id: int, enabled: bool) -> None:
    flag = 1 if enabled else 0
    n = p.getNumBodies()
    for i in range(n):
        other_id = p.getBodyUniqueId(i)
        if other_id != body_id:
            p.setCollisionFilterPair(body_id, other_id, -1, -1, flag)


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


def reset_cube_velocity(obj: int) -> None:
    p.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

def pickup_cube(cubeid: int, robotid: int, obj_dicts: dict) -> None:
    sim_cfg = cfg["simulation"]
    move_steps = cfg["motion"]["move_steps"]
    time_step = sim_cfg["time_step"]

    # 当前位姿
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cubeid)
    robot_pos, _ = p.getBasePositionAndOrientation(robotid)

    # 目标：机器人正上方（留一点间隙）
    robot_aabb_min, robot_aabb_max = p.getAABB(robotid)
    cube_aabb_min, cube_aabb_max = p.getAABB(cubeid)
    cube_half_h = 0.5 * (cube_aabb_max[2] - cube_aabb_min[2])
    gap = 0.01
    target_pos = [robot_pos[0], robot_pos[1], robot_aabb_max[2] + cube_half_h + gap]

    # 计算场景最高点，构造安全高度（避免路径穿过其他物体）
    max_top = -1e9
    n = p.getNumBodies()
    for i in range(n):
        bid = p.getBodyUniqueId(i)
        if bid == cubeid:
            continue
        _, aabb_max = p.getAABB(bid)
        if aabb_max[2] > max_top:
            max_top = aabb_max[2]

    clearance = 0.05
    safe_z = max(max_top + cube_half_h + clearance, cube_pos[2], target_pos[2])

    # 根据高度差决定路径顺序
    z_diff = abs(cube_pos[2] - target_pos[2])
    z_threshold = 0.3

    # 移动期间禁用与所有物体碰撞，保证“不会与任何物体发生碰撞”
    _set_collision_with_all(cubeid, enabled=False)

    if z_diff > z_threshold:
        # 先Z后XY再Z
        wp1 = [cube_pos[0], cube_pos[1], safe_z]
        wp2 = [target_pos[0], target_pos[1], safe_z]
        wp3 = target_pos
        move_tar(cubeid, wp1, cube_orn, max(1, move_steps // 3), time_step)
        move_tar(cubeid, wp2, cube_orn, max(1, move_steps // 3), time_step)
        move_tar(cubeid, wp3, cube_orn, max(1, move_steps - 2 * (move_steps // 3)), time_step)
    else:
        # 先XY后Z
        wp1 = [target_pos[0], target_pos[1], cube_pos[2]]
        wp2 = target_pos
        move_tar(cubeid, wp1, cube_orn, max(1, move_steps // 2), time_step)
        move_tar(cubeid, wp2, cube_orn, max(1, move_steps - (move_steps // 2)), time_step)

    reset_cube_velocity(cubeid)
    _set_collision_with_all(cubeid, enabled=True)


def drop_cube(top_cube, robotid, obj_dicts) -> None:

    sim_cfg = cfg["simulation"]
    _, robot_orn = p.getBasePositionAndOrientation(robotid)
    # Disable collisions while extracting the top cube so lower cubes remain undisturbed.
    set_cube_collisions(top_cube, robotid, obj_dicts["plane"], enabled=False)
    move_dir(
        top_cube,
        robot_orn,
        cfg["motion"]["x_offset"],
        cfg["motion"]["move_steps"],
        sim_cfg["time_step"],
    )

    # Re-enable dynamics and collisions so the extracted cube can fall onto the plane.
    set_cube_collisions(top_cube, robotid, obj_dicts["plane"], enabled=True)
    reset_cube_velocity(top_cube)

def try_glue(body_a, body_b):

    contacts = p.getContactPoints(bodyA=body_a, bodyB=body_b)

    # 用当前世界位姿计算 parent->child 的相对位姿，保证“当前姿态下粘住”
    pos_a, orn_a = p.getBasePositionAndOrientation(body_a)
    pos_b, orn_b = p.getBasePositionAndOrientation(body_b)

    inv_a_pos, inv_a_orn = p.invertTransform(pos_a, orn_a)
    child_pos_in_a, child_orn_in_a = p.multiplyTransforms(inv_a_pos, inv_a_orn, pos_b, orn_b)

    glue_cid = p.createConstraint(
        parentBodyUniqueId=body_a,
        parentLinkIndex=-1,
        childBodyUniqueId=body_b,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        # Use current relative transform as the parent frame so the constraint
        # starts with near-zero position/orientation error.
        parentFramePosition=child_pos_in_a,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=child_orn_in_a,
        childFrameOrientation=[0, 0, 0, 1],
    )

    # Prevent contact solver and fixed-constraint solver from fighting.
    p.setCollisionFilterPair(body_a, body_b, -1, -1, 0)
    p.changeConstraint(glue_cid, maxForce=200)
    return glue_cid

def unglue(glue_cid: int | None, body_a: int | None = None, body_b: int | None = None) -> None:
    if glue_cid is None:
        return
    p.removeConstraint(glue_cid)
    if body_a is not None and body_b is not None:
        p.setCollisionFilterPair(body_a, body_b, -1, -1, 1)
