from __future__ import annotations
import pybullet as p
from python_motion_planning import *
from config_loader import load_config
from control.move import goto, move_dir, move_tar
from control.plan import Grid, motionplan

cfg = load_config()
scaling_factor = cfg["simulation"]["scaling_factor"]


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



def reset_cube_velocity(obj: int) -> None:
    p.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

def pickup_cube(cubeid: int, robotid: int) -> None:
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

class MoveToTargetTask:
    def __init__(self, cube_stacks, cube_picked, delta_per_step=0.0005):
        self.reached = False
        self.stepsize = delta_per_step

        self.map = Grid(bounds=[[0, 1600], [0, 1600]])
        self.map.fill_boundary_with_obstacles()
        stacks_iter = cube_stacks.values() if hasattr(cube_stacks, "values") else cube_stacks
        for cubes in stacks_iter:
            for cube in cubes:
                if cube_picked.get(cube, True):
                    continue
                pos = p.getBasePositionAndOrientation(cube)[0]
                aabb_min, aabb_max = p.getAABB(cube)
                size = [
                    aabb_max[0] - aabb_min[0],  # x 方向长度
                    aabb_max[1] - aabb_min[1],  # y 方向长度
                    aabb_max[2] - aabb_min[2],  # z 方向长度
                ]
                x_min = int(round((pos[0] - size[0]/2) * scaling_factor))
                x_max = int(round((pos[0] + size[0]/2) * scaling_factor))
                y_min = int(round((pos[1] - size[1]/2) * scaling_factor))
                y_max = int(round((pos[1] + size[1]/2) * scaling_factor))
                print(f"Marking grid cells from ({x_min}, {y_min}) to ({x_max}, {y_max}) as obstacles")
                    
                self.map.type_map[x_min:x_max+1, y_min:y_max+1] = TYPES.OBSTACLE

        self.map.inflate_obstacles(radius=15)    

    def setup(self, target_pos, robot_id):
        self.target_pos = target_pos
        self.reached = False
        self.tragetory = motionplan(robot_id, self.map, self.target_pos)
        self.current_i = 0
        # Path planning is 2D, so keep the robot at its current height.
        self.cruise_z = p.getBasePositionAndOrientation(robot_id)[0][2]

    def reset(self, t0: float):
        self.reached = False

    def begin(self, robot_id: int) -> None:
        if not self.tragetory:
            print(f"Warning: Empty trajectory, target {self.target_pos} unreachable")
            self.reached = True
            return
            
        while not self.reached:
            current_pos = p.getBasePositionAndOrientation(robot_id)[0]
            target_2d = self.tragetory[self.current_i]
            # Expand 2D waypoint to 3D while preserving robot base height.
            target_3d = [target_2d[0], target_2d[1], self.cruise_z]
            
            # Reach check in XY only; Z is intentionally held constant.
            if all(abs(current_pos[i] - target_3d[i]) < 0.01 for i in range(2)):
                self.current_i += 1
                if self.current_i >= len(self.tragetory):
                    self.reached = True
            else:
                while not goto(robot_id, target_3d, speed=self.stepsize)[0]:
                    p.stepSimulation()

        return
