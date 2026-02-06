from coppeliasim_zmqremoteapi_client import *

# create a client to connect to zmqRemoteApi server:
# (creation arguments can specify different host/port,
# defaults are host='localhost', port=23000)
client = RemoteAPIClient()

# get a remote object:
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()

# call API function:
h = sim.getObject('/Floor')
print(h)

sim.stopSimulation()
print('Program ended')
