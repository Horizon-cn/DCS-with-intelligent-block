import utils
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from sim.scene import Scene
from control.tasks import SimpleMoveTask, MoveToTargetTask
from control.plan import motionplan
from control.move import goto
from config import cfg

def main():

    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + cfg["sim"]["scene_path"])

    robot_paths = utils.get_all_robot_paths(cfg)
    block_paths = utils.get_all_block_paths(cfg)

    task = SimpleMoveTask(cfg["sim"]["switch_period_s"], cfg["sim"]["delta_per_step"])
    UR5_1 = sim.getObject(robot_paths["UR5_1"])
    delta_per_step = cfg["sim"]["delta_per_step"]

    blocks = {name: sim.getObject(path) for name, path in block_paths.items()}

    client.setStepping(True)

    sim.startSimulation()
    try:
        t0 = sim.getSimulationTime()
        task.reset(t0)
        sim.setObjectPosition(UR5_1, -1, [5, 5, 0])  

        for k in range(cfg["run"]["max_steps"]):
            t, axis, p = task.update(sim, UR5_1)

            if (k % cfg["run"]["print_every"]) == 0:
                print(f"[k={k}] t={t:.2f}s axis={axis} pos={p}")

            client.step()

        task = MoveToTargetTask(sim, blocks, cfg=cfg, delta_per_step=delta_per_step)
        task.setup([5, 5, 0], sim, UR5_1)
        task.begin(client, sim, UR5_1)

        task.setup([7.75, 1.5, 0], sim, UR5_1)
        task.begin(client, sim, UR5_1)

        time.sleep(1)  # 延迟1秒

        task.setup([7.75, 1, 0], sim, UR5_1)
        task.begin(client, sim, UR5_1)

        time.sleep(1)  # 延迟1秒

        task.setup([4, 4, 0], sim, UR5_1)
        task.begin(client, sim, UR5_1)

    except KeyboardInterrupt:
        pass
    finally:
        sim.stopSimulation()

if __name__ == "__main__":
    main()
