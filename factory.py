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


def create_cube_shapes(cube_cfg: dict) -> tuple[int, int]:
    return _create_mesh_shape_pair(cube_cfg, "cube_scale")

def create_cube(
    visual_shape_id: int,
    collision_shape_id: int,
    cube_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
    base_orientation: tuple[float, float, float, float] | None = None,
) -> int:
    return _create_body(
        visual_shape_id,
        collision_shape_id,
        cube_cfg,
        base_position,
        use_maximal_coordinates,
        base_orientation=base_orientation,
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
            create_cube(
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
    base_platform_heading_dir: str = "+X"
    end_platform_heading_dir: str = "+X"
    fixed_heading_dir: str = "+X"
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

    def _find_link_index_by_name(self, link_name: str) -> int:
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            child_link_name = p.getJointInfo(self.robot_id, joint_idx)[12].decode("utf-8")
            if child_link_name == link_name:
                return joint_idx
        raise ValueError(f"Link {link_name!r} not found for robot {self.robot_id}")

    def __post_init__(self) -> None:
        self.END_PLATFORM_LINK = self._find_link_index_by_name("end_platform")
        _, self.base_platform_orientation = p.getBasePositionAndOrientation(self.robot_id)
        end_state = p.getLinkState(self.robot_id, self.END_PLATFORM_LINK)
        if end_state is None:
            raise ValueError(
                f"Failed to get link state for end_platform link index {self.END_PLATFORM_LINK}"
            )
        self.end_platform_orientation = end_state[1]
        self.base_platform_heading_dir = self._heading_dir_from_orientation(self.base_platform_orientation)
        self.end_platform_heading_dir = self._end_heading_from_base_heading(self.base_platform_heading_dir)
        self.fixed_heading_dir = self.base_platform_heading_dir

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

    def _closest_axis_label(self, v, allowed: tuple[str, ...] | None = None) -> str:
        allowed_labels = allowed if allowed is not None else tuple(self.FACE_NORM.keys())
        best_label = allowed_labels[0]
        best_dot = -1e9
        for label in allowed_labels:
            axis = self.FACE_NORM[label]
            dot = self._dot(v, axis)
            if dot > best_dot:
                best_dot = dot
                best_label = label
        return best_label

    def _heading_dir_from_orientation(self, orn, allowed: tuple[str, ...] | None = None) -> str:
        rot = p.getMatrixFromQuaternion(orn)
        x_axis = (rot[0], rot[3], rot[6])  # local +X in world
        return self._closest_axis_label(x_axis, allowed=allowed)

    @staticmethod
    def _end_heading_from_base_heading(base_heading: str) -> str:
        """Map base heading to end heading: invert Y/Z, keep X unchanged."""
        opposite_yz = {
            "+Y": "-Y",
            "-Y": "+Y",
            "+Z": "-Z",
            "-Z": "+Z",
        }
        return opposite_yz.get(base_heading, base_heading)

    def planner_start_heading_dir(self, node) -> str:
        tangent_axes = tuple(
            label for label, axis in self.FACE_NORM.items() if abs(self._dot(axis, self.FACE_NORM[node.face_dir])) < 1e-9
        )
        platform_orn = (
            self.base_platform_orientation
            if self.fixed_platform == "base_platform"
            else self.end_platform_orientation
        )
        return self._heading_dir_from_orientation(platform_orn, allowed=tangent_axes)

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

    def _platform_contact_sign(self, platform_name: str) -> float:
        # base_platform contacts with local -Z; end_platform contacts with local +Z.
        return -1.0 if platform_name == "base_platform" else 1.0

    def _platform_orientation_for_face(self, face_dir: str, platform_name: str, heading_dir: str | None = None):
        """World-frame quaternion with platform contact side on face_dir and +X on heading_dir."""
        n = self.FACE_NORM[face_dir]
        contact_sign = self._platform_contact_sign(platform_name)
        z_axis = (n[0] / contact_sign, n[1] / contact_sign, n[2] / contact_sign)
        if heading_dir is None:
            if face_dir in ("+Z", "-Z"):
                x_axis = (1.0, 0.0, 0.0)
            elif face_dir in ("+X", "-X"):
                x_axis = (0.0, 1.0, 0.0)
            else:
                x_axis = (1.0, 0.0, 0.0)
        else:
            x_axis = self.FACE_NORM[heading_dir]
            if abs(self._dot(x_axis, z_axis)) > 1e-9:
                raise ValueError(
                    f"Heading {heading_dir} is not tangent to face {face_dir} for platform {platform_name}"
                )

        y_axis = self._normalize(self._cross(z_axis, x_axis))
        x_axis = self._normalize(self._cross(y_axis, z_axis))
        return self._quat_from_axes(x_axis, y_axis, z_axis)

    def _node_face_midpoint(self, node):
        n = self.FACE_NORM[node.face_dir]
        return (node.pos[0] + 0.5 * n[0], node.pos[1] + 0.5 * n[1], node.pos[2] + 0.5 * n[2])

    def _platform_center_for_node_contact(self, node, platform_orn, platform_name: str):
        """Return the platform link center for a surface-contact pose.

        The node defines the target surface midpoint. The platform is placed so its
        contact face coincides with that surface. The base platform uses local `-Z`
        as its contact side; the end platform uses local `+Z`.
        """
        face_mid = self._node_face_midpoint(node)
        rot = p.getMatrixFromQuaternion(platform_orn)
        z_axis = (rot[2], rot[5], rot[8])  # local +Z in world
        center_offset = -self._platform_contact_sign(platform_name) * self.PLATFORM_HALF_THICKNESS
        return (
            face_mid[0] + z_axis[0] * center_offset+0.5,
            face_mid[1] + z_axis[1] * center_offset+0.5,
            face_mid[2] + z_axis[2] * center_offset,
        )

    def _platform_center_from_contact_point(self, contact_point, platform_orn, platform_name: str):
        """Convert a world-space contact point to platform link center pose."""
        rot = p.getMatrixFromQuaternion(platform_orn)
        z_axis = (rot[2], rot[5], rot[8])  # local +Z in world
        center_offset = -self._platform_contact_sign(platform_name) * self.PLATFORM_HALF_THICKNESS
        return (
            float(contact_point[0]) + z_axis[0] * center_offset+0.5,
            float(contact_point[1]) + z_axis[1] * center_offset+0.5,
            float(contact_point[2]) + z_axis[2] * center_offset,
        )

    def _avoid_negative_face_cylinder(self, point, face_dir: str, face_pos, radius: float = 0.4):
        """Shift point outside the cylinder along the +face_dir axis from face_pos."""
        axis_dir = self.FACE_NORM[face_dir]
        v = (point[0] - face_pos[0], point[1] - face_pos[1], point[2] - face_pos[2])
        axial = self._dot(v, axis_dir)

        # Only constrain points in the +face_dir half-line.
        if axial < 0.0:
            return point

        proj = (axis_dir[0] * axial, axis_dir[1] * axial, axis_dir[2] * axial)
        perp = (v[0] - proj[0], v[1] - proj[1], v[2] - proj[2])
        dist = self._norm(perp)
        if dist >= radius:
            return point

        if dist < 1e-9:
            ref = (1.0, 0.0, 0.0) if abs(axis_dir[0]) < 0.9 else (0.0, 1.0, 0.0)
            perp = self._normalize(self._cross(axis_dir, ref))
        else:
            perp = (perp[0] / dist, perp[1] / dist, perp[2] / dist)

        new_v = (
            proj[0] + perp[0] * radius,
            proj[1] + perp[1] * radius,
            proj[2] + perp[2] * radius,
        )
        return (face_pos[0] + new_v[0], face_pos[1] + new_v[1], face_pos[2] + new_v[2])

    def _resample_polyline_waypoints(
        self,
        points,
        waypoint_count: int = 8,
        face_dir: str | None = None,
        face_pos=None,
        avoid_radius: float = 0.4,
        min_spacing: float = 0.08,
    ):
        """Resample a polyline to a fixed number of waypoints, including endpoints."""
        def _nudge_inside_voxel(pt, clearance: float = 0.1):
            def _adjust_axis(v):
                base = math.floor(v)
                frac = v - base
                if frac < clearance:
                    return base + clearance
                if frac > 1.0 - clearance:
                    return base + (1.0 - clearance)
                return v

            return (
                _adjust_axis(pt[0]),
                _adjust_axis(pt[1]),
                _adjust_axis(pt[2]),
            )

        def _resample(points_in):
            pts = [tuple(float(v) for v in pt) for pt in points_in]
            if len(pts) <= 1 or waypoint_count <= 1:
                return pts

            seg_lengths = []
            total_len = 0.0
            for i in range(len(pts) - 1):
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
                dz = pts[i + 1][2] - pts[i][2]
                seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                seg_lengths.append(seg_len)
                total_len += seg_len

            if total_len < 1e-9:
                return [pts[0]] * waypoint_count

            targets = [total_len * i / (waypoint_count - 1) for i in range(waypoint_count)]
            out = [pts[0]]
            seg_idx = 0
            traversed = 0.0

            for target_dist in targets[1:-1]:
                while seg_idx < len(seg_lengths) - 1 and traversed + seg_lengths[seg_idx] < target_dist:
                    traversed += seg_lengths[seg_idx]
                    seg_idx += 1

                seg_len = max(seg_lengths[seg_idx], 1e-9)
                t = (target_dist - traversed) / seg_len
                p0 = pts[seg_idx]
                p1 = pts[seg_idx + 1]
                out.append(
                    (
                        p0[0] + (p1[0] - p0[0]) * t,
                        p0[1] + (p1[1] - p0[1]) * t,
                        p0[2] + (p1[2] - p0[2]) * t,
                    )
                )

            out.append(pts[-1])
            return out

        out = _resample(points)

        if face_dir is None or face_pos is None:
            return [_nudge_inside_voxel(pt) for pt in out]

        adjusted = [
            self._avoid_negative_face_cylinder(pt, face_dir, face_pos, radius=avoid_radius)
            for pt in out
        ]
        adjusted = [_nudge_inside_voxel(pt) for pt in adjusted]

        if min_spacing <= 0.0:
            return adjusted

        total_len = 0.0
        for i in range(len(adjusted) - 1):
            dx = adjusted[i + 1][0] - adjusted[i][0]
            dy = adjusted[i + 1][1] - adjusted[i][1]
            dz = adjusted[i + 1][2] - adjusted[i][2]
            total_len += math.sqrt(dx * dx + dy * dy + dz * dz)

        if total_len < min_spacing * max(1, waypoint_count - 1):
            return adjusted

        return [_nudge_inside_voxel(pt) for pt in _resample(adjusted)]

    def platform_contact_point(self, platform_name: str):
        """Return the current world-space contact point of the named platform."""
        link_id = self._platform_link(platform_name)
        link_pos, link_orn = self._link_pose(link_id)
        rot = p.getMatrixFromQuaternion(link_orn)
        z_axis = (rot[2], rot[5], rot[8])  # local +Z in world
        contact_offset = self._platform_contact_sign(platform_name) * self.PLATFORM_HALF_THICKNESS
        return (
            float(link_pos[0]) + z_axis[0] * contact_offset,
            float(link_pos[1]) + z_axis[1] * contact_offset,
            float(link_pos[2]) + z_axis[2] * contact_offset,
        )

    def current_moving_platform_contact_point(self):
        """Return the current contact point of the platform that will move next."""
        return self.platform_contact_point(self._other_platform(self.fixed_platform))

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

    def _movable_joints(self) -> list[int]:
        joints = []
        for j in range(p.getNumJoints(self.robot_id)):
            jt = p.getJointInfo(self.robot_id, j)[2]
            if jt == p.JOINT_REVOLUTE or jt == p.JOINT_PRISMATIC:
                joints.append(j)
        return joints

    def _ik_joint_limit_kwargs(self) -> dict:
        """Build joint limit arguments for PyBullet IK from the current URDF."""
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = []

        for joint_id in self._movable_joints():
            joint_info = p.getJointInfo(self.robot_id, joint_id)
            lower = float(joint_info[8])
            upper = float(joint_info[9])

            # Fall back to a wide range if the URDF does not provide a valid bounded interval.
            if upper < lower:
                lower = -2.0 * math.pi
                upper = 2.0 * math.pi

            lower_limits.append(lower)
            upper_limits.append(upper)
            joint_ranges.append(max(1e-6, upper - lower))
            rest_poses.append(float(p.getJointState(self.robot_id, joint_id)[0]))

        return {
            "lowerLimits": lower_limits,
            "upperLimits": upper_limits,
            "jointRanges": joint_ranges,
            "restPoses": rest_poses,
        }

    def _calculate_ik_for_platform_target(self, target_link: int, target_pos, target_orn):
        ik_kwargs = self._ik_joint_limit_kwargs()
        if target_link != self.BASE_PLATFORM_LINK:
            return p.calculateInverseKinematics(
                self.robot_id,
                target_link,
                target_pos,
                target_orn,
                **ik_kwargs,
                maxNumIterations=200,
                residualThreshold=1e-5,
            )

        fixed_end_pos, fixed_end_orn = self._link_pose(self.END_PLATFORM_LINK)
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        joint_states = [
            (j, p.getJointState(self.robot_id, j)[0])
            for j in range(p.getNumJoints(self.robot_id))
        ]

        p.resetBasePositionAndOrientation(self.robot_id, target_pos, target_orn)
        ik = p.calculateInverseKinematics(
            self.robot_id,
            self.END_PLATFORM_LINK,
            fixed_end_pos,
            fixed_end_orn,
            **ik_kwargs,
            maxNumIterations=500,
            residualThreshold=1e-5,
        )
        p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)
        for j, q in joint_states:
            p.resetJointState(self.robot_id, j, q)
        return ik

    def _smooth_apply_ik(
        self,
        target_link: int,
        target_pos,
        target_orn,
        steps: int = 180,
        smooth: bool = True,
        active_joint: int | None = None,
    ) -> None:
        ik = self._calculate_ik_for_platform_target(target_link, target_pos, target_orn)
        revolute_joints = self._movable_joints()

        cur = {j: p.getJointState(self.robot_id, j)[0] for j in revolute_joints}
        sim_cfg = cfg["simulation"]
        dt = 1/9600
        max_force = 900000000
        pos_tol = 0.005
        orn_tol = 0.035
        joint_tol = 0.01
        max_hold_steps = 60

        if active_joint is not None and active_joint not in cur:
            raise ValueError(f"active_joint {active_joint} is not a movable joint")

        def _joint_target_for_step(joint_id: int, interpolated_target: float, final_target: float, final: bool) -> float:
            if active_joint is None or joint_id == active_joint:
                return final_target if final else interpolated_target
            return cur[joint_id]

        for i in range(steps):
            t = (i + 1) / max(1, steps)
            s = t * t * (3.0 - 2.0 * t) if smooth else t
            for j, tj in zip(revolute_joints, ik):
                q = cur[j] + (tj - cur[j]) * s
                p.setJointMotorControl2(
                    self.robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=_joint_target_for_step(j, q, tj, final=False),
                    force=max_force,
                )
            p.stepSimulation()
            time.sleep(dt)

        # Keep driving the final IK target until the link actually converges.
        # This prevents motion from stopping early when fixed interpolation steps
        # are not sufficient to physically settle at the goal.
        for _ in range(max_hold_steps):
            for j, tj in zip(revolute_joints, ik):
                p.setJointMotorControl2(
                    self.robot_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=_joint_target_for_step(j, tj, tj, final=True),
                    force=max_force,
                )
            p.stepSimulation()
            time.sleep(dt)

            cur_pos, cur_orn = self._link_pose(target_link)
            pos_err = math.sqrt(
                (cur_pos[0] - target_pos[0]) ** 2
                + (cur_pos[1] - target_pos[1]) ** 2
                + (cur_pos[2] - target_pos[2]) ** 2
            )
            dot_q = abs(
                cur_orn[0] * target_orn[0]
                + cur_orn[1] * target_orn[1]
                + cur_orn[2] * target_orn[2]
                + cur_orn[3] * target_orn[3]
            )
            dot_q = max(-1.0, min(1.0, dot_q))
            orn_err = 2.0 * math.acos(dot_q)
            joint_err = 0.0
            for j, tj in zip(revolute_joints, ik):
                target_q = tj if active_joint is None or j == active_joint else cur[j]
                joint_err = max(joint_err, abs(p.getJointState(self.robot_id, j)[0] - target_q))

            if pos_err <= pos_tol and orn_err <= orn_tol and joint_err <= joint_tol:
                break

    def _compute_target_orientation(
        self,
        from_node,
        to_node,
        fixed_platform_pos,
        moving_platform_name: str,
        target_heading_dir: str | None = None,
    ):
        return self._platform_orientation_for_face(
            to_node.face_dir,
            moving_platform_name,
            heading_dir=target_heading_dir,
        )

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

    def step_forward(
        self,
        from_node,
        to_node,
        prev_node=None,
        spatial_path=None,
        target_heading_dir: str | None = None,
        target_fixed_platform: str | None = None,
    ) -> None:
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

        fixed_pos, fixed_orn = self._link_pose(fixed_link)
        fixed_contact_pos = self._platform_center_for_node_contact(from_node, fixed_orn, fixed_name)

        # lock fixed platform to current node plane at start-state.
        if self.fixed_cid is not None:
            p.removeConstraint(self.fixed_cid)
            self.fixed_cid = None
        if self.fixed_anchor_id is not None:
            p.removeBody(self.fixed_anchor_id)
            self.fixed_anchor_id = None
        self.fixed_anchor_id = self._create_anchor(fixed_contact_pos, fixed_orn)
        self.fixed_cid = self._create_lock_to_anchor(fixed_link, self.fixed_anchor_id)

        target_moving_orn = self._compute_target_orientation(
            from_node,
            to_node,
            fixed_contact_pos,
            moving_name,
            target_heading_dir=target_heading_dir,
        )
        target_moving_pos = self._platform_center_for_node_contact(to_node, target_moving_orn, moving_name)
        print(f"Moving {moving_name} from {fixed_name} contact at {fixed_contact_pos} to target node at {to_node.pos} with orientation {target_moving_orn}")

        # If planner provides a feasible shell/surface path, follow intermediate
        # contact waypoints to reduce large IK jumps.
        if spatial_path and len(spatial_path) > 2:
            face_pos = self._node_face_midpoint(from_node)
            path_points = self._resample_polyline_waypoints(
                spatial_path,
                waypoint_count=8,
                face_dir=from_node.face_dir,
                face_pos=face_pos,
                avoid_radius=0.4,
            )
            print(f"Following spatial path with {len(path_points)} waypoints for smoother motion")
            inner_points = path_points[1:-1]

            for contact_pt in inner_points:
                waypoint_pos = self._platform_center_from_contact_point(
                    contact_pt,
                    target_moving_orn,
                    moving_name,
                )
                self._smooth_apply_ik(
                    moving_link,
                    waypoint_pos,
                    target_moving_orn,
                    steps=160,
                    smooth=False,
                )

        self._smooth_apply_ik(moving_link, target_moving_pos, target_moving_orn, steps=160)

        ik = self._calculate_ik_for_platform_target(moving_link, target_moving_pos, target_moving_orn)
        revolute_joints = self._movable_joints()
        for j, tj in zip(revolute_joints, ik):
           p.resetJointState(self.robot_id, j, tj)
        if moving_link == self.BASE_PLATFORM_LINK:
           p.resetBasePositionAndOrientation(self.robot_id, target_moving_pos, target_moving_orn)
           p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # end-state: moving platform also locked on target node plane.
        if self.moving_cid is not None:
            p.removeConstraint(self.moving_cid)
            self.moving_cid = None
        if self.moving_anchor_id is not None:
            p.removeBody(self.moving_anchor_id)
            self.moving_anchor_id = None
        self.moving_anchor_id = self._create_anchor(target_moving_pos, target_moving_orn)
        self.moving_cid = self._create_lock_to_anchor(moving_link, self.moving_anchor_id)

        # The moved platform becomes the next support; release the previous support.
        if self.fixed_cid is not None:
            p.removeConstraint(self.fixed_cid)
        if self.fixed_anchor_id is not None:
            p.removeBody(self.fixed_anchor_id)
        self.fixed_cid = self.moving_cid
        self.fixed_anchor_id = self.moving_anchor_id
        self.moving_cid = None
        self.moving_anchor_id = None

        # Cache orientations for next-step orientation logic.
        base_pos, base_orn = self._link_pose(self.BASE_PLATFORM_LINK)
        end_pos, end_orn = self._link_pose(self.END_PLATFORM_LINK)
        self.base_platform_orientation = base_orn
        self.end_platform_orientation = end_orn
        self.base_platform_heading_dir = self._heading_dir_from_orientation(base_orn)
        self.end_platform_heading_dir = self._end_heading_from_base_heading(self.base_platform_heading_dir)

        # Next step swaps fixed platform (new support platform is the one just moved).
        self.fixed_platform = moving_name
        if target_heading_dir is not None:
            self.fixed_heading_dir = target_heading_dir
            if moving_name == "base_platform":
                self.base_platform_heading_dir = target_heading_dir
            else:
                self.end_platform_heading_dir = target_heading_dir
        else:
            self.fixed_heading_dir = (
                self.base_platform_heading_dir
                if self.fixed_platform == "base_platform"
                else self.end_platform_heading_dir
            )
        if target_fixed_platform is not None and target_fixed_platform != self.fixed_platform:
            raise ValueError(
                f"Planner/execution fixed platform mismatch: expected {target_fixed_platform}, got {self.fixed_platform}"
            )
