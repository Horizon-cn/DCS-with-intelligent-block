class Scene:
    def __init__(self, sim, object_path, scene_path: str):
        self.sim = sim
        self.obj = sim.getObject(object_path)  
        self.sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + scene_path)

    def time(self) -> float:
        return self.sim.getSimulationTime()

    def dt(self) -> float:
        return self.sim.getSimulationTimeStep()

    def get_pos(self):
        return self.sim.getObjectPosition(self.obj, -1)

    def set_pos(self, p):
        self.sim.setObjectPosition(self.obj, -1, p)

    def get_orientation(self):
        return self.sim.getObjectOrientation(self.obj, -1)
    
    def set_orientation(self, o):
        self.sim.setObjectOrientation(self.obj, -1, o)