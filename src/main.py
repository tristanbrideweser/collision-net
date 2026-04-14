import pybullet as p
import argparse
import time
from src.env.env import create_env, reset_env, settle_objects
from src.env.obstacles import spawn_clutter
from src.planner.rrt_connect import RRTConnect
from src.planner.collision_detector import init_collision_checker, update_obstacles
from src.planner.kinematics import get_goal_config

def main(args):
    dof = 7
    client = p.connect(p.GUI)
    plane_id, table_id, panda_id = create_env()
    object_types = [p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_SPHERE]
    num_objects = 6
    obj_ids = spawn_clutter(num_objects=num_objects,
                            object_types=object_types,
                            size_min=0.01,
                            size_max=0.05,
                            base_mass=0.01)
    settle_objects()
    obstacle_ids = [plane_id, table_id] + obj_ids
    init_collision_checker(panda_id=panda_id, obstacle_ids=obstacle_ids)
    # settle_objects()

    start = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    goal_pos = [1.1, -0.4, 0.75]
    goal = get_goal_config(panda_id=panda_id, target_pos=goal_pos)

    # CHANGE HERE
    use_nn_collision = False

    print("Planning path...")
    path = RRTConnect(start, goal, use_nn_collision)

    if path:
        print(f"path found with {len(path)} nodes")

        while True:
            for waypoint in path:
                for i in range(dof):
                    p.resetJointState(panda_id, i, waypoint[i])
                time.sleep(0.5)
            time.sleep(2)
    else:
        print("failed to find valid path")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    #max_steps = parser.add_argument('--max-steps', int=100)
    
    main(args)