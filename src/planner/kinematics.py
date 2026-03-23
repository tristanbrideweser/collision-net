import pybullet as p

def get_goal_config(panda_id, target_pos):
    joint_poses = p.calculateInverseKinematics(
        panda_id,
        11, 
        target_pos, 
        p.getQuaternionFromEuler([0, 3.14, 0])
    )

    return list(joint_poses[:7])