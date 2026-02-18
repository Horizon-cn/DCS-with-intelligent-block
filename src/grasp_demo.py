import math
from control.tasks import SimpleMoveTask, MoveToTargetTask
from config import cfg
import utils

# This is a threaded script, and is just an example.
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.require('sim')
sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + cfg["sim"]["scene_path"])

def enable_ik(enable):
    if enable:
        sim.setObjectMatrix(ikTarget, -1, sim.getObjectMatrix(ikTip, -1))
        for handle in jointHandles:
            sim.setJointMode(handle, sim.jointmode_ik, 1)
        sim.setExplicitHandling(ikGroupHandle, 0)
    else:
        sim.setExplicitHandling(ikGroupHandle, 1)
        for handle in jointHandles:
            sim.setJointMode(handle, sim.jointmode_dynamic, 0)


def set_gripper_data(open_gripper, velocity=0.11, force=20):
    if not open_gripper:
        velocity = -velocity
    data = sim.packFloatTable([velocity, force])
    sim.setStringSignal(f"{modelName}_rg2GripperData", data)


# Initialize some values.
block_paths = utils.get_all_block_paths(cfg)
jointHandles = [sim.getObjectHandle(f"UR5_joint{i}") for i in range(1, 7)]
ikTip = sim.getObject("/UR5_1/UR5_ikTip")
ikTarget = sim.getObject("/UR5_1/UR5_ikTarget")
modelBase = sim.getObject("/UR5_1")
modelName = sim.getObjectName(modelBase)
blocks = {name: sim.getObject(path) for name, path in block_paths.items()}

# Set up some of the RML vectors.
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

initialConfig = [0, 0, 0, 0, 0, 0]
pickConfig = [
    -70.1 * deg2rad,
    18.85 * deg2rad,
    93.18 * deg2rad,
    68.02 * deg2rad,
    109.9 * deg2rad,
    90 * deg2rad,
]
dropConfig1 = [
    -183.34 * deg2rad,
    14.76 * deg2rad,
    78.26 * deg2rad,
    -2.98 * deg2rad,
    -90.02 * deg2rad,
    86.63 * deg2rad,
]
dropConfig2 = [
    -197.6 * deg2rad,
    14.76 * deg2rad,
    78.26 * deg2rad,
    -2.98 * deg2rad,
    -90.02 * deg2rad,
    72.38 * deg2rad,
]
dropConfig3 = [
    -192.1 * deg2rad,
    3.76 * deg2rad,
    91.16 * deg2rad,
    -4.9 * deg2rad,
    -90.02 * deg2rad,
    -12.13 * deg2rad,
]
dropConfig4 = [
    -189.38 * deg2rad,
    24.94 * deg2rad,
    64.36 * deg2rad,
    0.75 * deg2rad,
    -90.02 * deg2rad,
    -9.41 * deg2rad,
]

dropConfigs = [dropConfig1, dropConfig2, dropConfig3, dropConfig4]
dropConfigIndex = 0
droppedPartsCnt = 0

task = SimpleMoveTask(cfg["sim"]["switch_period_s"], -cfg["sim"]["delta_per_step2"])
sim.startSimulation()

t0 = sim.getSimulationTime()
task.reset(t0)

for k in range(cfg["run"]["max_steps"]):
    t, axis, p = task.update_g(sim, ikTarget)

    if (k % cfg["run"]["print_every"]) == 0:
        print(f"[k={k}] t={t:.2f}s axis={axis} pos={p}")

    client.step()
task2 = MoveToTargetTask(sim, blocks, cfg=cfg)
task2.setup([5, 5, 0.5], sim, ikTarget)
task2.begin(client, sim, ikTarget)
task2.setup([4.87, 4.86, 0.7], sim, ikTarget)
task2.begin(client, sim, ikTarget)
task2.setup([4.96, 4.79, 0.67], sim, ikTarget)
task2.begin(client, sim, ikTarget)
task2.setup([5.03, 5, 1], sim, ikTarget)
task2.begin(client, sim, ikTarget)
print("done")

while True:
    client.step()

sim.stopSimulation()


set_gripper_data(True)
sim.setInt32Param(sim.intparam_current_page, 0)

while droppedPartsCnt < 6:

    sim.rmlMoveToJointPositions(
        jointHandles,
        -1,
        currentVel,
        currentAccel,
        maxVel,
        maxAccel,
        maxJerk,
        pickConfig,
        targetVel,
    )

    sim.setInt32Param(sim.intparam_current_page, 1)
    pos = sim.getObjectPosition(ikTip, -1)
    quat = sim.getObjectQuaternion(ikTip, -1)

    sim.rmlMoveToPosition(
        ikTarget,
        -1,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [pos[0] + 0.105, pos[1], pos[2]],
        quat,
        None,
    )

    set_gripper_data(False)
    sim.wait(0.5)

    sim.rmlMoveToPosition(
        ikTarget,
        -1,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [pos[0], pos[1] - 0.2, pos[2] + 0.2],
        quat,
        None,
    )

    enable_ik(False)
    sim.setInt32Param(sim.intparam_current_page, 0)

    sim.rmlMoveToJointPositions(
        jointHandles,
        -1,
        currentVel,
        currentAccel,
        maxVel,
        maxAccel,
        maxJerk,
        dropConfigs[dropConfigIndex],
        targetVel,
    )

    sim.setInt32Param(sim.intparam_current_page, 2)
    enable_ik(True)
    pos = sim.getObjectPosition(ikTip, -1)
    quat = sim.getObjectQuaternion(ikTip, -1)

    sim.rmlMoveToPosition(
        ikTarget,
        -1,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        [pos[0], pos[1], 0.025 + 0.05 * math.floor(0.1 + droppedPartsCnt / 2)],
        quat,
        None,
    )

    set_gripper_data(True)
    sim.wait(0.5)

    sim.rmlMoveToPosition(
        ikTarget,
        -1,
        -1,
        None,
        None,
        ikMaxVel,
        ikMaxAccel,
        ikMaxJerk,
        pos,
        quat,
        None,
    )

    enable_ik(False)
    sim.setInt32Param(sim.intparam_current_page, 0)
    dropConfigIndex += 1
    if dropConfigIndex >= len(dropConfigs):
        dropConfigIndex = 0

    droppedPartsCnt += 1

sim.rmlMoveToJointPositions(
    jointHandles,
    -1,
    currentVel,
    currentAccel,
    maxVel,
    maxAccel,
    maxJerk,
    initialConfig,
    targetVel,
)
sim.stopSimulation()