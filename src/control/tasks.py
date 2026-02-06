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

        p = sim.getObjectPosition(obj, -1)
        p[self.i] += self.delta
        sim.setObjectPosition(obj, -1, p)

        return t, self.i, p
