from src.env.obstacles import add_wall
import pybullet as p
import pybullet_data
import time

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
panda_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0.0, 0, 0.625],
    useFixedBase=True
)

add_wall((1.0, 0.0, 0.625))

while True:
    p.stepSimulation()
    time.sleep(1./240.)