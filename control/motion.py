from __future__ import annotations
import pybullet as p
import numpy as np
from typing import Dict
from python_motion_planning import *
from config_loader import load_config
from control.move import goto, move_dir, move_tar
from control.plan import Grid, motionplan
from control.dstar_surface_3d import DStarLiteSurface3D, Node, NORM, FACES

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
    @staticmethod
    def _iter_stacks(cube_stacks):
        """Yield stack lists from dict/list/2D-list containers."""
        if hasattr(cube_stacks, "values"):
            items = cube_stacks.values()
        else:
            items = cube_stacks

        for item in items:
            if isinstance(item, list):
                if not item:
                    continue
                if all(isinstance(cube_id, int) for cube_id in item):
                    yield item
                else:
                    yield from MoveToTargetTask._iter_stacks(item)

    def __init__(self, cube_stacks, cube_picked, delta_per_step=0.0005):
        self.reached = False
        self.stepsize = delta_per_step

        self.map = Grid(bounds=[[0, 1600], [0, 1600]])
        self.map.fill_boundary_with_obstacles()
        for cubes in self._iter_stacks(cube_stacks):
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
                #print(f"Marking grid cells from ({x_min}, {y_min}) to ({x_max}, {y_max}) as obstacles")
                    
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


class DynamicMoveToTargetTask:
    """
    支持 D* Lite 动态replanning的3D移动任务。
    在执行期间监测占用栅格变化，使用增量更新而不是重新规划。
    """
    def __init__(self, occ: np.ndarray, size_xyz: tuple, cube_stacks, cube_picked, delta_per_step: float = 0.002):
        """
        Args:
            occ: 3D occupancy array (X, Y, Z)
            size_xyz: (X, Y, Z) grid dimensions
            cube_stacks: cube position tracking
            cube_picked: cube pick status tracking
            delta_per_step: movement speed per simulation step
        """
        self.occ = occ  # Reference to occupancy grid (shared, monitored for changes)
        self.size_xyz = size_xyz
        self.cube_stacks = cube_stacks
        self.cube_picked = cube_picked
        self.stepsize = delta_per_step
        
        # Track previous occupancy for change detection
        self.prev_occ = np.copy(occ)
        
        # D* Lite planner (initialized in setup())
        self.planner = None
        self.path = []
        self.current_i = 0
        self.reached = False
        self.cruise_z = 0
        
        # Replanning config
        self.replan_interval = 10  # Check for map changes every N steps
        self.step_count = 0
        
    def detect_map_changes(self) -> Dict[tuple[int,int,int], int]:
        """
        Detect which voxels changed between current and previous occupancy.
        Returns dict mapping changed voxel coordinates to their new occupancy values.
        """
        changed = {}
        for x in range(self.size_xyz[0]):
            for y in range(self.size_xyz[1]):
                for z in range(self.size_xyz[2]):
                    if self.occ[x, y, z] != self.prev_occ[x, y, z]:
                        new_val = int(self.occ[x, y, z])
                        changed[(x, y, z)] = new_val
                        self.prev_occ[x, y, z] = self.occ[x, y, z]
        return changed
    
    def setup(self, start_pos: tuple[float,float,float], goal_pos: tuple[float,float,float], robot_id: int):
        """
        Initialize planner with start and goal nodes on obstacle surfaces.
        
        Args:
            start_pos: robot current position (x, y, z)
            goal_pos: target position (x, y, z)
            robot_id: PyBullet robot ID
        """
        self.target_pos = goal_pos
        self.reached = False
        self.current_i = 0
        self.cruise_z = p.getBasePositionAndOrientation(robot_id)[0][2]
        self.step_count = 0
        self.robot_id = robot_id
        
        # Find valid start and goal nodes (free cell centers adjacent to obstacles)
        start_node = self._find_closest_node(start_pos, robot_id)
        goal_node = self._find_closest_node(goal_pos, robot_id)
        
        if start_node is None or goal_node is None:
            print(f"Warning: Could not find valid start/goal nodes. Start={start_node}, Goal={goal_node}")
            self.reached = True
            return
        
        # Initialize D* Lite planner
        try:
            self.planner = DStarLiteSurface3D(self.occ, self.size_xyz, start_node, goal_node)
            self.path = self.planner.plan(max_steps=100)
            
            if not self.path:
                print(f"Warning: No path found from {start_node} to {goal_node}")
                self.reached = True
        except Exception as e:
            print(f"Error initializing D* Lite planner: {e}")
            self.reached = True
    
    def _find_closest_node(self, pos: tuple[float,float,float], robot_id: int) -> Node | None:
        """
        Find closest free voxel to position that has an exposed obstacle face.
        """
        x, y, z = pos
        grid_x = int(round(x))
        grid_y = int(round(y))
        grid_z = int(round(z))
        
        # Search radius for finding valid node
        search_radius = 1
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    vx, vy, vz = grid_x + dx, grid_y + dy, grid_z + dz
                    
                    # Check if voxel is in bounds and free
                    if not (0 <= vx < self.size_xyz[0] and 0 <= vy < self.size_xyz[1] and 0 <= vz < self.size_xyz[2]):
                        continue
                    if self.occ[vx, vy, vz] != 0:
                        continue
                    
                    # Find exposed faces (adjacent to obstacles)
                    for face in FACES:
                        ox = vx + NORM[face][0]
                        oy = vy + NORM[face][1]
                        oz = vz + NORM[face][2]
                        
                        if 0 <= ox < self.size_xyz[0] and 0 <= oy < self.size_xyz[1] and 0 <= oz < self.size_xyz[2]:
                            if self.occ[ox, oy, oz] != 0:
                                # Found valid node
                                center = (vx + 0.5, vy + 0.5, vz + 0.5)
                                return Node(center, face)
        
        return None
    
    def begin(self) -> None:
        """
        Execute motion along planner path and periodically replan on occupancy updates.
        """
        if not self.planner:
            print("Planner not initialized")
            self.reached = True
            return

        # If setup path was not available, try extracting from current planner state.
        if not self.path:
            self.path = self._extract_path_from_planner()
            self.current_i = 0
            if not self.path:
                print("No initial path available")
                self.reached = True
                return

        self.reached = False
        _set_collision_with_all(self.robot_id, enabled=False)

        try:
            while not self.reached:
                if self.current_i >= len(self.path):
                    self.reached = True
                    break

                current_pos = p.getBasePositionAndOrientation(self.robot_id)[0]

                # Periodically detect occupancy updates and trigger one batch replan.
                self.step_count += 1
                if self.step_count % self.replan_interval == 0:
                    changed_voxels = self.detect_map_changes()
                    if changed_voxels:
                        print(
                            f"Detected {len(changed_voxels)} map changes at step {self.step_count}, replanning..."
                        )

                        # Keep D* start aligned with current robot position before replanning.
                        cur_node = self._find_closest_node(current_pos, self.robot_id)
                        if cur_node is None:
                            print("No valid current node for replanning")
                            self.reached = True
                            continue

                        try:
                            self.planner.move_start_to(cur_node)
                        except Exception as e:
                            print(f"Failed to move planner start to current node: {e}")
                            self.reached = True
                            continue

                        self.planner.buffer_updates(changed_voxels)
                        self.planner.apply_batch_updates(replan=True)

                        self.path = self._extract_path_from_planner()
                        self.current_i = 0

                        if not self.path:
                            print("No path available after replanning")
                            self.reached = True
                        continue

                waypoint = self.path[self.current_i].pos
                target_3d = [waypoint[0]+0.5, waypoint[1]+0.5, waypoint[2]]

                # Advance index only when current path point is reached.
                if all(abs(current_pos[i] - target_3d[i]) < 0.05 for i in range(3)):
                    self.current_i += 1
                    continue

                # Move to the current waypoint using smooth interpolation.
                dx = target_3d[0] - current_pos[0]
                dy = target_3d[1] - current_pos[1]
                dz = target_3d[2] - current_pos[2]
                distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                step_len = max(1e-4, float(self.stepsize))
                move_steps = max(1, int(np.ceil(distance / step_len * 0.5)))  # 加速：减半移动步数
                _, cur_orn = p.getBasePositionAndOrientation(self.robot_id)
                move_tar(
                    self.robot_id,
                    target_3d,
                    cur_orn,
                    move_steps,
                    cfg["simulation"]["time_step"],
                )
                self.current_i += 1
        finally:
            _set_collision_with_all(self.robot_id, enabled=True)

        return
    
    def _extract_path_from_planner(self) -> list[Node]:
        """
        Extract path from D* Lite planner using stateless method.
        Does NOT modify planner's internal state (start position, km).
        """
        if not self.planner:
            return []
        
        # Use stateless path extraction to avoid modifying planner state
        path = self.planner.extract_path_stateless(max_steps=200)
        return path


# ============================================================================
# USAGE EXAMPLE: How to use DynamicMoveToTargetTask with batch replanning
# ============================================================================
"""
Example in main.py:

    from control.motion import DynamicMoveToTargetTask
    
    # During main loop:
    # 1. Create task with occupancy grid and size
    task = DynamicMoveToTargetTask(
        occ=occ,  # numpy array (X,Y,Z)
        size_xyz=(X, Y, Z),
        cube_stacks=cube_stacks,
        cube_picked=cube_picked,
        delta_per_step=0.002  # movement speed
    )
    
    # 2. Set replanning frequency (default: 10 steps)
    task.replan_interval = 20  # Check map changes every 20 simulation steps
    
    # 3. Setup with start/goal positions
    task.setup(
        start_pos=(6.5, 1.0, 1.0),
        goal_pos=(10.0, 10.0, 0.0),
        robot_id=robot1_id
    )
    
    # 4. Execute with automatic replanning on map changes
    task.begin()
    
BENEFITS OF BATCH REPLANNING:
- Only ONE compute_shortest_path() call per check interval instead of one per voxel change
- Accumulate all occupancy changes then update affected nodes in one pass
- Scalable: efficient even if many voxels change simultaneously
- Deferred replanning: buffer updates, apply them, THEN replan once

D* LITE BATCH UPDATE API:
- planner.buffer_update(v, new_occ)     -- Buffer one voxel update
- planner.buffer_updates(dict)           -- Buffer multiple updates as dict
- planner.apply_batch_updates(replan=True) -- Apply all buffered updates and replan
- planner.clear_buffer()                 -- Discard buffered updates
- planner.get_buffer_size()              -- Get number of pending updates

ADVANCED: Custom replanning triggers
    if <custom_condition>:
        changed = task.detect_map_changes()
        if changed:
            task.planner.buffer_updates(changed)
            task.planner.apply_batch_updates(replan=True)
"""
