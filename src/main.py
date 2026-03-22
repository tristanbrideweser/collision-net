import pybullet as p
from src.env.env import create_env, reset_env

def main():
    client = p.connect(p.GUI)
    create_env()

    while True:
        p.stepSimulation()

if __name__ == "__main__":
    main()