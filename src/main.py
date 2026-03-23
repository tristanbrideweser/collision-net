import pybullet as p
import argparse
from src.env.env import create_env, reset_env

def main(args):
    client = p.connect(p.GUI)
    create_env()

    while True:
        p.stepSimulation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    #max_steps = parser.add_argument('--max-steps', int=100)
    
    main(args)