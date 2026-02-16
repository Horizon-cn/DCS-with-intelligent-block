from control.move import *
from control.plan import *
from config import scaling_factor


class SimpleMoveTask:

    def __init__(self, switch_period_s: float, delta_per_step: float):
        self.switch_period_s = switch_period_s
        self.delta = delta_per_step
        self.prev_t = 0.0
        self.i = 0
        self.count = 0

    def reset(self, t0: float):
        self.prev_t = t0
        self.i = 0

    def update(self, sim, obj):
        t = sim.getSimulationTime()

        if t - self.prev_t > self.switch_period_s:
            self.prev_t = t
            self.i = 1 - self.i
            if self.i == 0:
                self.delta = -self.delta

        p = step(sim, obj, self.i, self.delta)

        return t, self.i, p

class MoveToTargetTask:
    def __init__(self, sim, blocks, cfg=None):
        self.reached = False
        self.stepsize = cfg["sim"]["delta_per_step"]

        self.map = Grid(bounds=[[0, 400], [0, 400]])
        self.map.fill_boundary_with_obstacles()
        for name, block in blocks.items():
            pos = sim.getObjectPosition(block, -1)
            size = sim.getObjectSizeFactor(block) /2 
            x_min = int(round((pos[0] - size/2) * scaling_factor))
            x_max = int(round((pos[0] + size/2) * scaling_factor))
            y_min = int(round((pos[1] - size/2) * scaling_factor))
            y_max = int(round((pos[1] + size/2) * scaling_factor))
            print(f"Marking grid cells from ({x_min}, {y_min}) to ({x_max}, {y_max}) as obstacles")
            
            self.map.type_map[x_min:x_max+1, y_min:y_max+1] = TYPES.OBSTACLE
        self.map.inflate_obstacles(radius=7)


        


    def setup(self, target_pos, sim, obj):
        self.target_pos = target_pos
        self.reached = False
        self.tragetory = motionplan(sim, obj, self.map, self.target_pos)
        self.current_i = 0

    def reset(self, t0: float):
        self.reached = False

    def begin(self, client, sim, obj):
        if not self.tragetory:
            print(f"Warning: Empty trajectory, target {self.target_pos} unreachable")
            self.reached = True
            return
            
        while not self.reached:
            current_pos = sim.getObjectPosition(obj, -1)
            target_2d = self.tragetory[self.current_i]
            # 扩展为3D坐标（使用目标的z坐标）
            target_3d = [target_2d[0], target_2d[1], self.target_pos[2]]
            
            # 只比较x, y坐标（前2个维度）
            if all(abs(current_pos[i] - target_2d[i]) < 0.01 for i in range(2)):
                self.current_i += 1
                if self.current_i >= len(self.tragetory):
                    self.reached = True
            else:
                while not goto(sim, obj, target_3d, speed=self.stepsize)[0]:
                    client.step()

        return