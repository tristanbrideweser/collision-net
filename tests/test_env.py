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
# Get the Axis-Aligned Bounding Box (AABB)
# Returns: [min_x, min_y, min_z], [max_x, max_y, max_z]
min_aabb, max_aabb = p.getAABB(table_id)

# Calculate dimensions
width = max_aabb[0] - min_aabb[0]   # X-axis
length = max_aabb[1] - min_aabb[1]  # Y-axis
height = max_aabb[2] - min_aabb[2]  # Z-axis

print(f"Table dimensions: Width={width:.2f}, Length={length:.2f}, Height={height:.2f}")

while True:
    p.stepSimulation()
    time.sleep(1./240.)