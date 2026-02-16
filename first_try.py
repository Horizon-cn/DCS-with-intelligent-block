from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def main():
    client = RemoteAPIClient()  # 端口按你CoppeliaSim设置
    sim = client.require('sim')
    sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + '/messaging/movementViaRemoteApitest.ttt')

    # 关键：remote 端启用 stepping（由 Python 推进仿真）
    client.setStepping(True)

    # 只取一次句柄，不要每步 getObject（你脚本里每次取其实没必要）
    obj = sim.getObject('/blueRobot')

    prev_t = 0.0
    i = 0

    sim.startSimulation()
    try:
        # 你原脚本是“直到仿真停止”，remote 里通常用循环 + 手动停止
        while not sim.getSimulationStopping():
            t = sim.getSimulationTime()

            # 和你脚本一致：每 10 秒切一次轴
            if t - prev_t > 10.0:
                prev_t = t
                i = (i + 1) % 3

            p = sim.getObjectPosition(obj, -1)
            p[i] += 0.001
            sim.setObjectPosition(obj, -1, p)

            print(f"Simulation time: {t:.2f} [s]")

            # 等价于脚本里的 sim.step()
            client.step()

    except KeyboardInterrupt:
        # Ctrl+C 退出
        pass
    finally:
        sim.stopSimulation()

if __name__ == "__main__":
    main()

