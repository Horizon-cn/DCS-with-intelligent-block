from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import List, Tuple

import pybullet as p

from config_loader import load_config
from control.motion import MoveToTargetTask, drop_cube, pickup_cube, try_glue, unglue
from control.move import rotate_to

cfg = load_config()

def _create_mesh_shape_pair(mesh_cfg: dict, scale_key: str) -> tuple[int, int]:
    shared_kwargs = {
        "shapeType": p.GEOM_MESH,
        "fileName": mesh_cfg["mesh_file"],
        "meshScale": mesh_cfg[scale_key],
    }
    visual_shape_id = p.createVisualShape(
        rgbaColor=mesh_cfg["visual_rgba"],
        specularColor=mesh_cfg["visual_specular"],
        visualFramePosition=mesh_cfg["frame_shift"],
        **shared_kwargs,
    )
    collision_shape_id = p.createCollisionShape(
        collisionFramePosition=mesh_cfg["frame_shift"],
        **shared_kwargs,
    )
    return visual_shape_id, collision_shape_id

def _create_body(
    visual_shape_id: int,
    collision_shape_id: int,
    body_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
    base_orientation: tuple[float, float, float, float] | None = None,
) -> int:
    body_id = p.createMultiBody(
        baseMass=body_cfg["mass"],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        useMaximalCoordinates=use_maximal_coordinates,
        **({"baseOrientation": base_orientation} if base_orientation is not None else {}),
    )
    p.changeDynamics(body_id, -1, lateralFriction=body_cfg["lateral_friction"])
    return body_id


def create_cube_shapes_org(cube_cfg: dict) -> tuple[int, int]:
    return _create_mesh_shape_pair(cube_cfg, "cube_scale")

def create_cube_org(
    visual_shape_id: int,
    collision_shape_id: int,
    cube_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
) -> int:
    return _create_body(
        visual_shape_id,
        collision_shape_id,
        cube_cfg,
        base_position,
        use_maximal_coordinates,
    )

def create_cube_shapes(cube_cfg: dict) -> tuple[int, int]:
    return _create_mesh_shape_pair(cube_cfg, "cube_scale")

def create_cube(
    visual_shape_id: int,
    collision_shape_id: int,
    cube_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
) -> int:
    return _create_body(
        visual_shape_id,
        collision_shape_id,
        cube_cfg,
        base_position,
        use_maximal_coordinates,
        base_orientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
    )


def create_cube_stack(
    visual_shape_id: int,
    collision_shape_id: int,
    cube_cfg: dict,
    stack_cfg: dict,
    use_maximal_coordinates: bool,
    z: int | None = None,
    base_pos: List[float] | None = None,
) -> List[int]:
    cubes: List[int] = []
    count = int(z) if z is not None else stack_cfg["count"]
    base_pos = base_pos if base_pos is not None else stack_cfg["base_position"]
    z_spacing = stack_cfg["z_spacing"]

    for i in range(count):
        cubes.append(
            create_cube_org(
                visual_shape_id,
                collision_shape_id,
                cube_cfg,
                [base_pos[0], base_pos[1], base_pos[2] + z_spacing * i],
                use_maximal_coordinates,
            )
        )

    return cubes

def create_robot_shapes(robot_cfg: dict) -> tuple[int, int]:
    return _create_mesh_shape_pair(robot_cfg, "robot_scale")

def create_robot(
    visual_shape_id: int,
    collision_shape_id: int,
    robot_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
) -> int:
    return _create_body(
        visual_shape_id,
        collision_shape_id,
        robot_cfg,
        base_position,
        use_maximal_coordinates,
    )

@dataclass
class rob_info:
    robot_id: int
    cube_picked: dict[int, bool]
    has_load: bool = False
    load_cube_id: int | None = None
    glue_cid: int | None = None
    # Step-planning state: which platform is currently fixed to a node surface.
    fixed_platform: str = "base_platform"
    # Cached platform orientations (world frame quaternions).
    base_platform_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    end_platform_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    fixed_cid: int | None = None
    moving_cid: int | None = None
    fixed_anchor_id: int | None = None
    moving_anchor_id: int | None = None

    BASE_PLATFORM_LINK = -1
    END_PLATFORM_LINK = 9
    # new_robot.urdf platform box size is 0.4 x 0.4 x 0.1 (local Z thickness=0.1).
    PLATFORM_HALF_THICKNESS = 0.05
    FACE_NORM = {
        "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
    }

    def __post_init__(self) -> None:
        _, self.base_platform_orientation = p.getBasePositionAndOrientation(self.robot_id)
        end_state = p.getLinkState(self.robot_id, self.END_PLATFORM_LINK)
        self.end_platform_orientation = end_state[1]

    def _platform_link(self, name: str) -> int:
        return self.BASE_PLATFORM_LINK if name == "base_platform" else self.END_PLATFORM_LINK

    def _other_platform(self, name: str) -> str:
        return "end_platform" if name == "base_platform" else "base_platform"

    def _link_pose(self, link_id: int):
        if link_id == -1:
            return p.getBasePositionAndOrientation(self.robot_id)
        ls = p.getLinkState(self.robot_id, link_id)
        return ls[0], ls[1]

    @staticmethod
    def _dot(a, b) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _norm(v) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    def _normalize(self, v, eps: float = 1e-9):
        n = self._norm(v)
        if n < eps:
            return (0.0, 0.0, 0.0)
        return (v[0] / n, v[1] / n, v[2] / n)

    def _project_to_plane(self, v, n):
        dn = self._dot(v, n)
        return (v[0] - dn * n[0], v[1] - dn * n[1], v[2] - dn * n[2])

    def _quat_from_axes(self, x_axis, y_axis, z_axis):
        r00, r01, r02 = x_axis[0], y_axis[0], z_axis[0]
        r10, r11, r12 = x_axis[1], y_axis[1], z_axis[1]
        r20, r21, r22 = x_axis[2], y_axis[2], z_axis[2]

        tr = r00 + r11 + r22
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (r21 - r12) / s
            qy = (r02 - r20) / s
            qz = (r10 - r01) / s
        elif r00 > r11 and r00 > r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
            qw = (r21 - r12) / s
            qx = 0.25 * s
            qy = (r01 + r10) / s
            qz = (r02 + r20) / s
        elif r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
            qw = (r02 - r20) / s
            qx = (r01 + r10) / s
            qy = 0.25 * s
            qz = (r12 + r21) / s
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
            qw = (r10 - r01) / s
            qx = (r02 + r20) / s
            qy = (r12 + r21) / s
            qz = 0.25 * s
        return (qx, qy, qz, qw)

    def _node_face_midpoint(self, node):
        n = self.FACE_NORM[node.face_dir]
        return (node.pos[0] + 0.5 * n[0], node.pos[1] + 0.5 * n[1], node.pos[2] + 0.5 * n[2])

    def _platform_center_for_node_contact(self, node, platform_orn):
        """Return platform center so local -Z face contacts the node surface midpoint.

        This mirrors `_spawn_pose_from_start_node` orientation semantics where local
        `-Z` is aligned with the surface normal (`face_dir`).
        """
        face_mid = self._node_face_midpoint(node)
        rot = p.getMatrixFromQuaternion(platform_orn)
        z_axis = (rot[2], rot[5], rot[8])  # local +Z in world
        return (
            face_mid[0] + z_axis[0] * self.PLATFORM_HALF_THICKNESS,
            face_mid[1] + z_axis[1] * self.PLATFORM_HALF_THICKNESS,
            face_mid[2] + z_axis[2] * self.PLATFORM_HALF_THICKNESS,
        )

    def _create_anchor(self, pos, orn) -> int:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=1e-3)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=1e-3, rgbaColor=[1, 1, 0, 0])
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=orn,
        )

    def _create_lock_to_anchor(self, link_id: int, anchor_id: int, max_force: float = 1e6) -> int:
        link_pos, link_orn = self._link_pose(link_id)
        anc_pos, anc_orn = p.getBasePositionAndOrientation(anchor_id)
        inv_link_pos, inv_link_orn = p.invertTransform(link_pos, link_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_link_pos, inv_link_orn, anc_pos, anc_orn)
        cid = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=link_id,
            childBodyUniqueId=anchor_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        p.changeConstraint(cid, maxForce=max_force)
        return cid

    def _smooth_apply_ik(self, target_link: int, target_pos, target_orn, steps: int = 180) -> None:
        ik = p.calculateInverseKinematics(
            self.robot_id,
            target_link,
            target_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        revolute_joints = []
        for j in range(p.getNumJoints(self.robot_id)):
            jt = p.getJointInfo(self.robot_id, j)[2]
            if jt == p.JOINT_REVOLUTE or jt == p.JOINT_PRISMATIC:
                revolute_joints.append(j)

        cur = {j: p.getJointState(self.robot_id, j)[0] for j in revolute_joints}
        sim_cfg = cfg["simulation"]
        dt = sim_cfg["time_step"]
        max_force = 1200

        for i in range(steps):
            t = (i + 1) / max(1, steps)
            s = t * t * (3.0 - 2.0 * t)
            for j in revolute_joints:
                if j >= len(ik):
                    continue
                tj = ik[j]
                q = cur[j] + (tj - cur[j]) * s
                p.setJointMotorControl2(
                    self.robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=q,
                    force=max_force,
                )
            p.stepSimulation()
            time.sleep(dt)

    def _compute_target_orientation(self, from_node, to_node, fixed_platform_pos, moving_platform_name: str):
        n_from = self.FACE_NORM[from_node.face_dir]
        n_to = self.FACE_NORM[to_node.face_dir]
        from_mid = self._node_face_midpoint(from_node)
        to_mid = self._node_face_midpoint(to_node)

        # moving platform local +X in world; this defines "front".
        moving_q = self.base_platform_orientation if moving_platform_name == "base_platform" else self.end_platform_orientation
        rot = p.getMatrixFromQuaternion(moving_q)
        moving_front = (rot[0], rot[3], rot[6])
        moving_right = (rot[1], rot[4], rot[7])

        disp = (to_mid[0] - from_mid[0], to_mid[1] - from_mid[1], to_mid[2] - from_mid[2])
        disp_proj = self._project_to_plane(disp, n_from)
        front_proj = self._normalize(self._project_to_plane(moving_front, n_from))
        right_proj = self._normalize(self._project_to_plane(moving_right, n_from))
        fp = self._dot(disp_proj, front_proj)
        rp = self._dot(disp_proj, right_proj)

        to_fixed = (
            fixed_platform_pos[0] - to_mid[0],
            fixed_platform_pos[1] - to_mid[1],
            fixed_platform_pos[2] - to_mid[2],
        )
        toward_fixed_proj = self._normalize(self._project_to_plane(to_fixed, n_to))

        parallel = abs(abs(self._dot(n_from, n_to)) - 1.0) < 1e-6
        if parallel:
            # front/back/left/right -> point toward fixed platform.
            if abs(fp) < 1e-6 or abs(rp) < 1e-6:
                x_axis = toward_fixed_proj
            else:
                # right-front/right-back -> point left; left-front/left-back -> point right.
                right_on_to = self._normalize(self._project_to_plane(right_proj, n_to))
                if rp > 0:
                    x_axis = (-right_on_to[0], -right_on_to[1], -right_on_to[2])
                else:
                    x_axis = right_on_to
        else:
            # Perpendicular faces: orient by perpendicular direction, biased to face fixed platform.
            x_axis = self._cross(n_to, n_from)
            if self._dot(x_axis, to_fixed) < 0:
                x_axis = (-x_axis[0], -x_axis[1], -x_axis[2])
            x_axis = self._normalize(x_axis)
            if self._norm(x_axis) < 1e-8:
                x_axis = toward_fixed_proj

        if self._norm(x_axis) < 1e-8:
            fallback = (1.0, 0.0, 0.0)
            x_axis = self._normalize(self._project_to_plane(fallback, n_to))

        # Platform normal opposes surface normal to realize face contact.
        z_axis = (-n_to[0], -n_to[1], -n_to[2])
        y_axis = self._normalize(self._cross(z_axis, x_axis))
        x_axis = self._normalize(self._cross(y_axis, z_axis))

        return self._quat_from_axes(x_axis, y_axis, z_axis)

    def glue_to_cube(self, cube_id: int) -> None:
        self.glue_cid = try_glue(self.robot_id, cube_id)

    def pick_up(self, cube_id: int) -> None:
        self.has_load = True
        self.load_cube_id = cube_id
        pickup_cube(self.load_cube_id, self.robot_id)
        self.glue_to_cube(cube_id)
        self.cube_picked[self.load_cube_id] = True

    def drop(self, obj_dict: dict) -> None:
        if self.load_cube_id is None:
            return

        # Glue may not exist if the robot never contacted the cube.
        if self.glue_cid is not None:
            unglue(self.glue_cid, self.robot_id, self.load_cube_id)

        drop_cube(self.load_cube_id, self.robot_id, obj_dict)
        self.cube_picked[self.load_cube_id] = False
        self.has_load = False
        self.load_cube_id = None
        self.glue_cid = None
    
    def rotate(self, angle: float) -> None:
        rotate_to(self.robot_id, angle)

    def move_to(self, target_pos: List[float], cube_stacks, delta_per_step: float = 0.002) -> None:
        task = MoveToTargetTask(cube_stacks, self.cube_picked, delta_per_step=delta_per_step)
        task.setup(target_pos, self.robot_id)
        task.begin(self.robot_id)

    def step_forward(self, from_node, to_node) -> None:
        """
        Move one step on 3D path with dual-platform locking semantics.
        - First step defaults to fixed base platform.
        - Start/end states keep platforms locked to node surfaces via fixed constraints.
        - IK computes full robot terminal joint/link state.
        - Joint motion uses smooth interpolation.
        """
        fixed_name = self.fixed_platform
        moving_name = self._other_platform(fixed_name)
        fixed_link = self._platform_link(fixed_name)
        moving_link = self._platform_link(moving_name)

        from_pos = self._node_face_midpoint(from_node)

        fixed_pos, fixed_orn = self._link_pose(fixed_link)
        # lock fixed platform to current node plane at start-state.
        if self.fixed_cid is not None:
            p.removeConstraint(self.fixed_cid)
            self.fixed_cid = None
        if self.fixed_anchor_id is not None:
            p.removeBody(self.fixed_anchor_id)
            self.fixed_anchor_id = None

        self.fixed_anchor_id = self._create_anchor(from_pos, fixed_orn)
        self.fixed_cid = self._create_lock_to_anchor(fixed_link, self.fixed_anchor_id)

        target_moving_orn = self._compute_target_orientation(from_node, to_node, fixed_pos, moving_name)
        target_moving_pos = self._platform_center_for_node_contact(to_node, target_moving_orn)
        self._smooth_apply_ik(moving_link, target_moving_pos, target_moving_orn, steps=180)

        # end-state: moving platform also locked on target node plane.
        if self.moving_cid is not None:
            p.removeConstraint(self.moving_cid)
            self.moving_cid = None
        if self.moving_anchor_id is not None:
            p.removeBody(self.moving_anchor_id)
            self.moving_anchor_id = None

        self.moving_anchor_id = self._create_anchor(target_moving_pos, target_moving_orn)
        self.moving_cid = self._create_lock_to_anchor(moving_link, self.moving_anchor_id)

        # Cache orientations for next-step orientation logic.
        base_pos, base_orn = self._link_pose(self.BASE_PLATFORM_LINK)
        end_pos, end_orn = self._link_pose(self.END_PLATFORM_LINK)
        self.base_platform_orientation = base_orn
        self.end_platform_orientation = end_orn

        # Next step swaps fixed platform (new support platform is the one just moved).
        self.fixed_platform = moving_name
