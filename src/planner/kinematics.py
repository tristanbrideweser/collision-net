# src/planner/kinematics.py

import numpy as np
import pybullet as p

from src.planner.collision_detector import (
    init_collision_checker,
    in_collision,
    reset_collision_count,
    JOINT_LIMITS,
)

EE_LINK_INDEX = 11

JOINT_LIMITS_LOWER = [lo for lo, _ in JOINT_LIMITS]
JOINT_LIMITS_UPPER = [hi for _, hi in JOINT_LIMITS]
JOINT_RANGES = [hi - lo for lo, hi in JOINT_LIMITS]


def get_ee_position(panda_id, config, arm_indices, ee_link=EE_LINK_INDEX):
    """Set arm to config and return end-effector Cartesian position."""
    for joint_id, val in zip(arm_indices, config):
        p.resetJointState(panda_id, joint_id, val)
    ee_state = p.getLinkState(panda_id, ee_link)
    return list(ee_state[0])


def get_goal_config(panda_id, target_pos):
    """Simple IK without collision checking (legacy)."""
    joint_poses = p.calculateInverseKinematics(
        panda_id,
        EE_LINK_INDEX,
        target_pos,
        p.getQuaternionFromEuler([0, 3.14, 0]),
    )
    return list(joint_poses[:7])


def ee_to_config(panda_id, target_pos, arm_indices, table_id=None,
                 ee_link=EE_LINK_INDEX, max_attempts=100, pos_tol=0.05):
    """
    Collision-aware IK: try multiple random seeds and return the first
    joint config that (a) places the EE within `pos_tol` of `target_pos`
    and (b) is collision-free in an empty scene (table + self only).

    Args:
        panda_id:     PyBullet body ID of the Panda
        target_pos:   [x, y, z] desired EE position
        arm_indices:  list of revolute joint indices
        table_id:     PyBullet body ID of the table (for collision check).
                      If None, skips collision checking (legacy behavior).
        ee_link:      link index of the end-effector
        max_attempts: number of random IK seeds to try
        pos_tol:      max Euclidean distance between achieved and target EE pos

    Returns:
        list[float] joint config (7 values), or None if no valid solution found
    """
    # If no table_id provided, fall back to single-shot IK (no collision check)
    if table_id is None:
        joint_poses = p.calculateInverseKinematics(
            panda_id, ee_link, target_pos,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        return list(joint_poses[:7])

    # Set up collision checker against empty scene (table + self-collision only)
    init_collision_checker(panda_id=panda_id, obstacle_ids=[], table_id=table_id)

    for attempt in range(max_attempts):
        # Random seed so IK explores different solution branches
        seed = [np.random.uniform(lo, hi)
                for lo, hi in zip(JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)]

        for i, val in enumerate(seed):
            p.resetJointState(panda_id, i, val)

        joint_poses = p.calculateInverseKinematics(
            panda_id, ee_link, target_pos,
            lowerLimits=JOINT_LIMITS_LOWER,
            upperLimits=JOINT_LIMITS_UPPER,
            jointRanges=JOINT_RANGES,
            restPoses=seed,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        conf = list(joint_poses[:7])

        # Check collision (empty scene — just table + self)
        reset_collision_count()
        if in_collision(conf):
            continue

        # Verify EE actually reached the target
        for i, val in enumerate(conf):
            p.resetJointState(panda_id, i, val)
        ee_pos = np.array(p.getLinkState(panda_id, ee_link)[0])
        error = np.linalg.norm(ee_pos - np.array(target_pos))

        if error < pos_tol:
            reset_collision_count()
            return conf

    # No valid solution found
    return None

def get_arm_link_positions(panda_id, config, arm_indices):
    """Get positions of all arm links for clearance checking."""
    for joint_id, val in zip(arm_indices, config):
        p.resetJointState(panda_id, joint_id, val)
    
    positions = []
    for i in range(p.getNumJoints(panda_id)):
        link_state = p.getLinkState(panda_id, i)
        positions.append(list(link_state[0]))
    return positions