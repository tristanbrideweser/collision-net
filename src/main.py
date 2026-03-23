import pybullet as p
import argparse
from src.env.env import create_env, reset_env, settle_objects
from src.planner.rrt_connect import RRTConnect
from src.planner.collision_detector import init_collision_checker, update_obstacles

def main(args):
    client = p.connect(p.GUI)
    create_env()
    settle_objects()
    init_collision_checker()

    start = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    goal = [1.0, -0.5, 0.5, -1.5, 0.3, 1.2, 0.5]
    path = RRTConnect(start, goal)

    while True:
        p.stepSimulation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    #max_steps = parser.add_argument('--max-steps', int=100)
    
    main(args)