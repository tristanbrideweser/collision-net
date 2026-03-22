JOINT_LIMITS = [
    (-2.8973,  2.8973),  # joint 1
    (-1.7628,  1.7628),  # joint 2
    (-2.8973,  2.8973),  # joint 3
    (-3.0718, -0.0698),  # joint 4
    (-2.8973,  2.8973),  # joint 5
    (-0.0175,  3.7525),  # joint 6
    (-2.8973,  2.8973),  # joint 7
]

def in_collision(conf) -> bool:
    '''
    The collision function that the planner calls
    
    Args:
        conf: the configuration of the robot

    Returns:
        bool: True if the config is in collision
    '''
    # maybe check join limits??
    # check self collision
    # check obstacle collsion
    return True