import math
from control.tasks import SimpleMoveTask, MoveToTargetTask
from config import cfg
import utils

vel = 180
accel = 40
jerk = 80
currentVel = [0, 0, 0, 0, 0, 0, 0]
currentAccel = [0, 0, 0, 0, 0, 0, 0]
deg2rad = math.pi / 180
maxVel = [vel * deg2rad] * 6
maxAccel = [accel * deg2rad] * 6
maxJerk = [jerk * deg2rad] * 6
targetVel = [0, 0, 0, 0, 0, 0]

ikMaxVel = [0.4, 0.4, 0.4, 1.8]
ikMaxAccel = [0.8, 0.8, 0.8, 0.9]
ikMaxJerk = [0.6, 0.6, 0.6, 0.8]

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.require('sim')

cuboid_paths = utils.get_all_cuboid_paths(cfg)
#jointHandles = [sim.getObjectHandle(f"UR5_joint{i}") for i in range(1, 7)]

ikTip = sim.getObject("/UR5_1/UR5_ikTip")
ikTarget = sim.getObject("/UR5_1/UR5_ikTarget")
modelBase = sim.getObject("/UR5_1")
modelName = sim.getObjectName(modelBase)
cuboids = {name: sim.getObject(path) for name, path in cuboid_paths.items()}

def set_gripper_data(open_gripper, velocity=0.11, force=20):
    if not open_gripper:
        velocity = -velocity
    data = sim.packFloatTable([velocity, force])
    sim.setStringSignal(f"{modelName}_rg2GripperData", data)


def grasp_obj(sim, obj_name):
    obj_handle = cuboids[obj_name]
    obj_pos = sim.getObjectPosition(obj_handle, -1)
    me_pos = sim.getObjectPosition(modelBase, -1)
    relative_pos = [obj_pos[i] - me_pos[i] for i in range(3)]

    sim.rmlMoveToPosition(
            modelBase,
            -1,
            -1,
            None,
            None,
            ikMaxVel,
            ikMaxAccel,
            ikMaxJerk,
            [me_pos[0], me_pos[1], me_pos[2]],
            [0.0, 0.0, 1.0, 0.0],
            None,
        )

    set_gripper_data(True)
    sim.wait(2)

    sim.rmlMoveToPosition(
        ikTarget,
        modelBase,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [-relative_pos[0], -relative_pos[1], relative_pos[2]],
        [-0.5, 0.5, -0.5, -0.5],
        None,
    )

    set_gripper_data(False)
    sim.wait(2)

    sim.rmlMoveToPosition(
        ikTarget,
        modelBase,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [0, -0.1, 0.6],
        [0.7071, 0.0, 0.0, 0.7071],
        None,
    )

    sim.rmlMoveToPosition(
            modelBase,
            -1,
            -1,
            None,
            None,
            ikMaxVel,
            ikMaxAccel,
            ikMaxJerk,
            [me_pos[0], me_pos[1], me_pos[2]],
            [0.0, 0.0, 0, 1.0],
            None,
        )

def drop_obj(sim, target_pos):
    me_pos = sim.getObjectPosition(modelBase, -1)
    relative_pos = [target_pos[i] - me_pos[i] for i in range(3)]


    sim.rmlMoveToPosition(
        ikTarget,
        modelBase,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [relative_pos[0], relative_pos[1], relative_pos[2]],
        [-0.5, 0.5, -0.5, -0.5],
        None,
    )

    set_gripper_data(True)
    sim.wait(2)

    sim.rmlMoveToPosition(
        ikTarget,
        modelBase,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [0, -0.1, 0.6],
        [0.7071, 0.0, 0.0, 0.7071],
        None,
    )

    sim.rmlMoveToPosition(
        modelBase,
        -1,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [me_pos[0], me_pos[1], me_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
        None,
    )