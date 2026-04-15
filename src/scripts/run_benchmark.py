"""
Milestone 1 Benchmark: Run RRT-Connect with geometric collision checker
across sparse, medium, and dense clutter environments.

Usage:
    python run_benchmark.py                      # all densities, 3 paths each
    python run_benchmark.py --density sparse      # single density
    python run_benchmark.py --runs 5              # 5 paths per density
    python run_benchmark.py --headless            # no GUI
"""

import argparse
import time
import pybullet as p
import numpy as np

from src.env.env import create_env, reset_env, settle_objects
from src.env.obstacles import spawn_clutter
from src.planner.rrt_connect import RRTConnect, path_smoothing
from src.planner.collision_detector import (
    init_collision_checker,
    update_obstacles,
    in_collision,
)
from src.planner.kinematics import get_goal_config
from src.utils.logger import PlanningLogger
from src.utils.table import print_summary_table, print_per_run_table


DENSITY_MAP = {
    "sparse": 5,
    "medium": 15,
    "dense": 25,
}

# Define multiple start/goal pairs to vary the planning problems
START_CONFIGS = [
    [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
    [0.5, -0.6, 0.3, -2.0, 0.2, 1.3, 0.5],
    [-0.3, -0.9, -0.2, -2.5, -0.1, 1.8, 1.0],
]

GOAL_POSITIONS = [
    [0.5, 0.2, 0.75],
    [0.6, -0.2, 0.80],
    [0.4, 0.0, 0.70],
]

OBJECT_TYPES = [p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_SPHERE]


def get_args():
    parser = argparse.ArgumentParser(description="Milestone 1 Planning Benchmark")
    parser.add_argument(
        "--density",
        type=str,
        default=None,
        choices=["sparse", "medium", "dense"],
        help="Run a single density tier (default: run all)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of paths per density"
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--smooth", action="store_true", help="Apply path smoothing")
    parser.add_argument(
        "--log", type=str, default="planning_log.json", help="Log filename"
    )
    return parser.parse_args()


def run_single_plan(panda_id, obj_ids, table_id, start_conf, goal_pos,
                    logger, density, run_id, smooth=False):
    """Execute one planning run and log results."""

    init_collision_checker(panda_id=panda_id, obstacle_ids=obj_ids, table_id=table_id)

    goal_conf = get_goal_config(panda_id=panda_id, target_pos=goal_pos)

    # Check that start and goal are valid
    if in_collision(start_conf):
        print(f"  Run {run_id}: Start config is in collision, skipping")
        logger.start_run(density, DENSITY_MAP[density], run_id)
        logger.end_run(None, success=False)
        return None

    if in_collision(goal_conf):
        print(f"  Run {run_id}: Goal config is in collision, skipping")
        logger.start_run(density, DENSITY_MAP[density], run_id)
        logger.end_run(None, success=False)
        return None

    # Patch in_collision to count calls
    original_in_collision = in_collision.__wrapped__ if hasattr(in_collision, '__wrapped__') else None

    logger.start_run(density, DENSITY_MAP[density], run_id)
    print(f"  Run {run_id}: Planning from start to goal_pos={goal_pos}...")

    path = RRTConnect(start_conf, goal_conf)

    if path and smooth:
        path = path_smoothing(path)

    success = path is not None and len(path) > 0
    logger.end_run(path, success)

    if success:
        print(f"  Run {run_id}: Path found — {len(path)} waypoints, "
              f"{logger.runs[-1]['planning_time_s']:.2f}s")
    else:
        print(f"  Run {run_id}: FAILED to find path")

    return path


def visualize_path(panda_id, path, arm_indices, delay=0.05):
    """Animate the arm along the planned path in the GUI."""
    if path is None:
        return
    for waypoint in path:
        for joint_id, val in zip(arm_indices, waypoint):
            p.resetJointState(panda_id, joint_id, val)
        time.sleep(delay)


def main():
    args = get_args()
    logger = PlanningLogger()

    # Determine which densities to run
    if args.density:
        densities = [args.density]
    else:
        densities = ["sparse", "medium", "dense"]

    # Connect to PyBullet
    if args.headless:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)

    plane_id, table_id, panda_id = create_env()

    arm_indices = [
        i for i in range(p.getNumJoints(panda_id))
        if p.getJointInfo(panda_id, i)[2] == p.JOINT_REVOLUTE
    ]

    for density in densities:
        num_objects = DENSITY_MAP[density]
        print(f"\n{'='*50}")
        print(f"Density: {density} ({num_objects} objects)")
        print(f"{'='*50}")

        for run_id in range(args.runs):
            # Reset scene
            # Remove old objects if any
            obj_ids = spawn_clutter(
                num_objects=num_objects,
                object_types=OBJECT_TYPES,
                size_min=0.01,
                size_max=0.05,
                base_mass=0.01,
                robot_id=panda_id,
                start_pos=[0, 0, 0.6],
                goal_pos=GOAL_POSITIONS[run_id % len(GOAL_POSITIONS)],
            )
            settle_objects()

            start = START_CONFIGS[run_id % len(START_CONFIGS)]
            goal_pos = GOAL_POSITIONS[run_id % len(GOAL_POSITIONS)]

            path = run_single_plan(
                panda_id, obj_ids, table_id, start, goal_pos,
                logger, density, run_id, smooth=args.smooth,
            )

            # Visualize in GUI mode
            if not args.headless and path:
                visualize_path(panda_id, path, arm_indices)
                time.sleep(1.0)

    # Save and print results
    logger.save(args.log)

    print(f"\n")
    print_summary_table(logger.get_runs())
    print()
    print_per_run_table(logger.get_runs())


if __name__ == "__main__":
    main()