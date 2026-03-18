from __future__ import annotations
from typing import List
import pybullet as p
from dataclasses import dataclass
from control.motion import *
from control.move import *



def create_cube_shapes(cube_cfg: dict) -> tuple[int, int]:
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=cube_cfg["mesh_file"],
        rgbaColor=cube_cfg["visual_rgba"],
        specularColor=cube_cfg["visual_specular"],
        visualFramePosition=cube_cfg["frame_shift"],
        meshScale=cube_cfg["cube_scale"],
    )

    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=cube_cfg["mesh_file"],
        collisionFramePosition=cube_cfg["frame_shift"],
        meshScale=cube_cfg["cube_scale"],
    )

    return visual_shape_id, collision_shape_id

def create_cube(
    visual_shape_id: int,
    collision_shape_id: int,
    cube_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
) -> int:
    cube_id = p.createMultiBody(
        baseMass=cube_cfg["mass"],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        useMaximalCoordinates=use_maximal_coordinates,
    )
    p.changeDynamics(cube_id, -1, lateralFriction=cube_cfg["lateral_friction"])
    return cube_id


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
        cube_id = p.createMultiBody(
            baseMass=cube_cfg["mass"],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[base_pos[0], base_pos[1], base_pos[2] + z_spacing * i],
            useMaximalCoordinates=use_maximal_coordinates,
        )
        p.changeDynamics(cube_id, -1, lateralFriction=cube_cfg["lateral_friction"])
        cubes.append(cube_id)

    return cubes

def create_robot_shapes(robot_cfg: dict) -> tuple[int, int]:
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=robot_cfg["mesh_file"],
        rgbaColor=robot_cfg["visual_rgba"],
        specularColor=robot_cfg["visual_specular"],
        visualFramePosition=robot_cfg["frame_shift"],
        meshScale=robot_cfg["robot_scale"],
    )

    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=robot_cfg["mesh_file"],
        collisionFramePosition=robot_cfg["frame_shift"],
        meshScale=robot_cfg["robot_scale"],
    )

    return visual_shape_id, collision_shape_id

def create_robot(
    visual_shape_id: int,
    collision_shape_id: int,
    robot_cfg: dict,
    base_position: List[float],
    use_maximal_coordinates: bool,
) -> int:
    robot_id = p.createMultiBody(
        baseMass=robot_cfg["mass"],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        useMaximalCoordinates=use_maximal_coordinates,
    )
    p.changeDynamics(robot_id, -1, lateralFriction=robot_cfg["lateral_friction"])
    return robot_id

@dataclass
class rob_info:
    robot_id: int
    cube_picked: dict[int, bool]
    has_load: bool = False
    load_cube_id: int | None = None
    glue_cid: int | None = None

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
        