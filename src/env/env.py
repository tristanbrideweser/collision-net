# src/env/env.py

import random
import pybullet as p
import pybullet_data

def create_env(gui=True):
    """Initialize PyBullet with table and Panda arm."""
    if p.getConnectionInfo()['isConnected'] == 0:
        p.connect(p.GUI if gui else p.DIRECT)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
    panda_id = p.loadURDF("franka_panda/panda.urdf",
                          basePosition=[0.0, 0, 0.625],
                          useFixedBase=True)

    return plane_id, table_id, panda_id


def reset_env(panda_id, obj_ids):
    """Remove all clutter and reset the arm to home config."""
    for obj_id in obj_ids:
        p.removeBody(obj_id)

    home_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i, val in enumerate(home_config):
        p.resetJointState(panda_id, i, val)

    return []

def settle_objects(steps=500):
    """Step physics to let spawned objects settle on the table."""
    for _ in range(steps):
        p.stepSimulation()