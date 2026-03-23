import numpy as np
import pybullet as p

JOINT_LIMITS = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
]

# module-level state set by init_collision_checker
_panda_id = None
_obstacle_ids = []
_table_id = None
_num_joints = 7


def init_collision_checker(panda_id, obstacle_ids, table_id=None):
    """Call this once after setting up the environment."""
    global _panda_id, _obstacle_ids, _table_id
    _panda_id = panda_id
    _obstacle_ids = obstacle_ids
    _table_id = table_id


def update_obstacles(obstacle_ids):
    """Update obstacle list when scene changes."""
    global _obstacle_ids
    _obstacle_ids = obstacle_ids


def _set_config(conf):
    """Set the robot to a given joint configuration."""
    for i in range(_num_joints):
        p.resetJointState(_panda_id, i, conf[i])


def _check_joint_limits(conf) -> bool:
    """Returns True if any joint is out of limits."""
    for i, val in enumerate(conf):
        lo, hi = JOINT_LIMITS[i]
        if val < lo or val > hi:
            return True
    return False


def _check_self_collision() -> bool:
    """Returns True if the robot is in self-collision."""
    p.performCollisionDetection()
    contacts = p.getContactPoints(_panda_id, _panda_id)
    # filter out adjacent link contacts (they always touch)
    for c in contacts:
        link_a, link_b = c[3], c[4]
        if abs(link_a - link_b) > 1:
            return True
    return False


def _check_obstacle_collision() -> bool:
    """Returns True if the robot collides with any obstacle."""
    p.performCollisionDetection()
    for obs_id in _obstacle_ids:
        contacts = p.getContactPoints(_panda_id, obs_id)
        if len(contacts) > 0:
            return True
    # also check table if provided
    if _table_id is not None:
        contacts = p.getContactPoints(_panda_id, _table_id)
        if len(contacts) > 0:
            return True
    return False


def in_collision(conf) -> bool:
    """
    The collision function that the planner calls.

    Args:
        conf: the configuration of the robot

    Returns:
        bool: True if the config is in collision
    """
    if _panda_id is None:
        raise RuntimeError("Call init_collision_checker() first")

    # check joint limits
    if _check_joint_limits(conf):
        return True

    # set robot to config
    _set_config(conf)

    # check self collision
    if _check_self_collision():
        return True

    # check obstacle collision
    if _check_obstacle_collision():
        return True

    return False