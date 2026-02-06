import yaml
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from sim.scene import Scene
from control.tasks import SimpleMoveTask

def main():
    cfg = yaml.safe_load(open("../config/default.yaml", "r", encoding="utf-8"))

    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + cfg["sim"]["scene_path"])

    task = SimpleMoveTask(cfg["sim"]["switch_period_s"], cfg["sim"]["delta_per_step"])
    obj = sim.getObject(cfg["sim"]["object_path"])

    client.setStepping(True)

    sim.startSimulation()
    try:
        t0 = sim.getSimulationTime()
        task.reset(t0)

        for k in range(cfg["run"]["max_steps"]):
            t, axis, p = task.update(sim, obj)

            if (k % cfg["run"]["print_every"]) == 0:
                print(f"[k={k}] t={t:.2f}s axis={axis} pos={p}")

            client.step()

    except KeyboardInterrupt:
        pass
    finally:
        sim.stopSimulation()

if __name__ == "__main__":
    main()
