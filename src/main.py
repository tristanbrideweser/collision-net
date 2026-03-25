# src/main.py

"""
Usage:
    python main.py --density sparse                          # single run, GUI
    python main.py --density dense --num-paths 5 --log       # 5 runs, log results
    python main.py --density all --num-paths 3 --log --plot  # full benchmark
    python main.py --headless --density all --log --plot      # headless benchmark
"""

import argparse
import random
import time
import os
import json
import pybullet as p
import numpy as np

from src.env.env import create_env, reset_env, settle_objects
from src.env.obstacles import spawn_clutter, spawn_structured_clutter
from src.planner.rrt_connect import RRTConnect
from src.planner.collision_detector import (
    init_collision_checker,
    update_obstacles,
    in_collision,
    get_collision_count,
    reset_collision_count,
    JOINT_LIMITS,
)
from src.planner.kinematics import *


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DENSITY_MAP = {
    "sparse": 10,
    "interspersed": 20,
    "dense": 40,
}

START_CONFIGS = [
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    [0.8, -0.4, 0.6, -1.5, 0.5, 1.0, 0.3],
    [-0.8, -1.2, -0.5, -2.8, -0.4, 2.0, 1.2],
    [0.4, -0.3, 0.8, -1.2, 0.6, 0.8, 0.1],
    [-0.6, -1.0, -0.8, -2.6, -0.5, 2.2, 1.5],
]

GOAL_CONFIGS = [
    [2.0, -1.2, -1.5, -0.8, 1.5, 3.0, -1.0],
    [-1.5, -1.7, 1.5, -3.0, -1.2, 0.3, 2.0],
    [1.8, -0.1, -1.0, -0.5, 2.0, 2.5, -1.5],
    [-1.2, -1.5, 1.0, -3.0, -1.5, 3.2, 1.2],
    [1.8, -1.2, -1.5, -0.5, 1.5, 0.5, -1.5],
]

for i in range(min(len(START_CONFIGS), len(GOAL_CONFIGS))):
    s = np.array(START_CONFIGS[i])
    g = np.array(GOAL_CONFIGS[i])
    print(f"Pair {i}: distance = {np.linalg.norm(s - g):.2f}")

OBJECT_TYPES = [p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_SPHERE]

# end-effector link index for the Panda
EE_LINK_INDEX = 11


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="CS 558 Milestone 1")

    # environment
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--density", type=str, default="sparse",
                        choices=["sparse", "interspersed", "dense", "all"],
                        help="Clutter density or 'all' for full benchmark")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI")

    # planning
    parser.add_argument("--num-paths", type=int, default=3,
                        help="Number of paths per density")

    # logging
    parser.add_argument("--log", action="store_true",
                        help="Save planning metrics to JSON")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for log files")

    # plotting
    parser.add_argument("--plot", action="store_true",
                        help="Generate result figures")
    parser.add_argument("--fig-dir", type=str, default="figures",
                        help="Directory for figures")

    return parser.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_ee_position(panda_id, config, arm_indices):
    """Set arm to config and return end-effector Cartesian position."""
    for joint_id, val in zip(arm_indices, config):
        p.resetJointState(panda_id, joint_id, val)
    ee_state = p.getLinkState(panda_id, EE_LINK_INDEX)
    return list(ee_state[0])


def measure_collision_rate(n_samples=1000):
    """Sample random configs and return the fraction that collide.
    Must be called AFTER init_collision_checker()."""
    n_hit = sum(
        1 for _ in range(n_samples)
        if in_collision([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
    )
    return n_hit / n_samples


# ──────────────────────────────────────────────
# Planning
# ──────────────────────────────────────────────

def run_single_plan(panda_id, obj_ids, table_id, start_conf, goal_conf):
    """Execute one planning run with joint-space start and goal.
    Returns (path, metrics_dict)."""

    init_collision_checker(panda_id=panda_id, obstacle_ids=obj_ids, table_id=table_id)

    # plan
    reset_collision_count()
    t0 = time.time()
    path = RRTConnect(start_conf, goal_conf)
    elapsed = time.time() - t0
    checks = get_collision_count()

    # compute path length
    path_length = 0.0
    if path and len(path) > 1:
        for i in range(1, len(path)):
            path_length += np.linalg.norm(
                np.array(path[i]) - np.array(path[i - 1])
            )

    success = path is not None and len(path) > 0
    metrics = {
        "success": success,
        "reason": "ok" if success else "planner_failed",
        "planning_time_s": round(elapsed, 4),
        "path_length": round(path_length, 4),
        "num_waypoints": len(path) if path else 0,
        "collision_checks": checks,
    }

    return path, metrics


def visualize_path(panda_id, path, arm_indices, delay=0.05):
    """Animate the arm along the path in the GUI."""
    if path is None:
        return
    for waypoint in path:
        for joint_id, val in zip(arm_indices, waypoint):
            p.resetJointState(panda_id, joint_id, val)
        time.sleep(delay)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def save_log(runs, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, "planning_log.json")
    with open(filepath, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\nLog saved to {filepath}")


def print_summary(runs):
    densities = sorted(set(r["density"] for r in runs))

    header = (
        f"{'Density':<14} "
        f"{'Runs':>5} "
        f"{'Success%':>9} "
        f"{'Avg Time(s)':>12} "
        f"{'Avg Path Len':>13} "
        f"{'Avg Col Chks':>13}"
    )
    print("\n" + "=" * len(header))
    print("RRT-Connect Results (Geometric Collision Checker)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for density in densities:
        d_runs = [r for r in runs if r["density"] == density]
        total = len(d_runs)
        successes = [r for r in d_runs if r["success"]]
        success_rate = len(successes) / total * 100 if total > 0 else 0

        if successes:
            avg_time = np.mean([r["planning_time_s"] for r in successes])
            std_time = np.std([r["planning_time_s"] for r in successes])
            avg_path = np.mean([r["path_length"] for r in successes])
            avg_checks = np.mean([r["collision_checks"] for r in successes])
        else:
            avg_time = std_time = avg_path = avg_checks = 0

        print(
            f"{density:<14} "
            f"{total:>5} "
            f"{success_rate:>8.1f}% "
            f"{avg_time:>7.2f}±{std_time:<4.2f} "
            f"{avg_path:>13.2f} "
            f"{avg_checks:>13.0f}"
        )

    print("=" * len(header))

    # detail table
    print(f"\n{'Density':<14} {'Run':>4} {'Status':>8} {'Time(s)':>8} "
          f"{'Path Len':>9} {'Waypts':>7} {'Col Chks':>9} {'Scene Try':>10} {'Col Rate':>9}")
    print("-" * 90)
    for r in runs:
        status = "OK" if r["success"] else "FAIL"
        col_rate = r.get("collision_rate", 0)
        print(
            f"{r['density']:<14} "
            f"{r['run_id']:>4} "
            f"{status:>8} "
            f"{r['planning_time_s']:>8.3f} "
            f"{r['path_length']:>9.3f} "
            f"{r['num_waypoints']:>7} "
            f"{r['collision_checks']:>9} "
            f"{r.get('scene_attempts', 0):>10} "
            f"{col_rate:>8.1f}%"
        )


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def generate_plots(runs, fig_dir):
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    density_order = ["sparse", "interspersed", "dense"]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    groups = {}
    for r in runs:
        groups.setdefault(r["density"], []).append(r)

    present = [d for d in density_order if d in groups]

    # 1) Planning time
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [np.mean([r["planning_time_s"] for r in groups[d] if r["success"]] or [0]) for d in present]
    stds = [np.std([r["planning_time_s"] for r in groups[d] if r["success"]] or [0]) for d in present]
    ax.bar(present, means, yerr=stds, capsize=5,
           color=[colors[density_order.index(d)] for d in present], alpha=0.85)
    ax.set_ylabel("Planning Time (s)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("RRT-Connect Planning Time vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "planning_time.png"), dpi=150)
    plt.close()

    # 2) Success rate
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = [sum(1 for r in groups[d] if r["success"]) / len(groups[d]) * 100 for d in present]
    ax.bar(present, rates,
           color=[colors[density_order.index(d)] for d in present], alpha=0.85)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Planning Success Rate vs Clutter Density")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "success_rate.png"), dpi=150)
    plt.close()

    # 3) Collision checks (box plot — successful runs only)
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[r["collision_checks"] for r in groups[d] if r["success"]] or [0] for d in present]
    bp = ax.boxplot(data, labels=present, patch_artist=True)
    for patch, d in zip(bp["boxes"], present):
        patch.set_facecolor(colors[density_order.index(d)])
        patch.set_alpha(0.6)
    ax.set_ylabel("Collision Checks")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Collision Checks per Plan vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "collision_checks.png"), dpi=150)
    plt.close()

    # 4) Path length
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [np.mean([r["path_length"] for r in groups[d] if r["success"]] or [0]) for d in present]
    stds = [np.std([r["path_length"] for r in groups[d] if r["success"]] or [0]) for d in present]
    ax.bar(present, means, yerr=stds, capsize=5,
           color=[colors[density_order.index(d)] for d in present], alpha=0.85)
    ax.set_ylabel("Path Length (joint-space)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Path Length vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "path_length.png"), dpi=150)
    plt.close()

    # 5) C-space collision rate per density
    fig, ax = plt.subplots(figsize=(6, 4))
    col_rates = [[r["collision_rate"] for r in groups[d] if "collision_rate" in r] or [0] for d in present]
    means = [np.mean(cr) for cr in col_rates]
    stds = [np.std(cr) for cr in col_rates]
    ax.bar(present, means, yerr=stds, capsize=5,
           color=[colors[density_order.index(d)] for d in present], alpha=0.85)
    ax.set_ylabel("C-space Collision Rate (%)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Random Config Collision Rate vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "collision_rate.png"), dpi=150)
    plt.close()

    print(f"Figures saved to {fig_dir}/")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.density == "all":
        densities = ["sparse", "interspersed", "dense"]
    else:
        densities = [args.density]

    if args.headless:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)

    plane_id, table_id, panda_id = create_env()

    arm_indices = [
        i for i in range(p.getNumJoints(panda_id))
        if p.getJointInfo(panda_id, i)[2] == p.JOINT_REVOLUTE
    ]

    all_runs = []
    obj_ids = []

    for density in densities:
        num_objects = DENSITY_MAP[density]
        print(f"\n{'='*50}")
        print(f"  {density.upper()} — {num_objects} objects")
        print(f"{'='*50}")

        for run_id in range(args.num_paths):
            print(f"\n  Run {run_id + 1}/{args.num_paths}")

            start = START_CONFIGS[run_id % len(START_CONFIGS)]
            goal = GOAL_CONFIGS[run_id % len(GOAL_CONFIGS)]

            # compute EE positions for spawn clearance
            start_ee_pos = get_ee_position(panda_id, start, arm_indices)
            goal_ee_pos = get_ee_position(panda_id, goal, arm_indices)

            # compute ALL link positions for both configs — these become
            # exclusion zones so obstacles don't spawn on the arm
            start_links = get_arm_link_positions(panda_id, start, arm_indices)
            goal_links = get_arm_link_positions(panda_id, goal, arm_indices)
            exclusion_points = start_links + goal_links

            valid = False
            scene_attempts = 0

            while not valid and scene_attempts < 20:
                # clear previous scene
                for obj_id in obj_ids:
                    p.removeBody(obj_id)
                obj_ids = []

                # reset arm to start
                for joint_id, val in zip(arm_indices, start):
                    p.resetJointState(panda_id, joint_id, val)

                # spawn fresh clutter with link-level exclusion zones
                obj_ids = spawn_structured_clutter(
                    num_objects=num_objects,
                    robot_id=panda_id,
                    start_pos=start_ee_pos,
                    goal_pos=goal_ee_pos,
                    difficulty=density,
                    exclusion_points=exclusion_points,
                )
                settle_objects()

                # validate start and goal configs
                init_collision_checker(panda_id=panda_id, obstacle_ids=obj_ids, table_id=table_id)

                start_ok = not in_collision(start)
                goal_ok = not in_collision(goal)

                if start_ok and goal_ok:
                    valid = True
                else:
                    scene_attempts += 1
                    reason = "start" if not start_ok else "goal"
                    print(f"    Scene invalid ({reason} collision), "
                          f"regenerating... (attempt {scene_attempts})")

            if not valid:
                print("    Could not generate valid scene after 20 attempts")
                metrics = {
                    "success": False,
                    "reason": "no_valid_scene",
                    "planning_time_s": 0,
                    "path_length": 0,
                    "num_waypoints": 0,
                    "collision_checks": 0,
                    "collision_rate": 0,
                    "scene_attempts": scene_attempts,
                }
                path = None
            else:
                print(f"    Valid scene found (attempt {scene_attempts + 1})")

                # ── C-space collision rate diagnostic ──
                # Measures what fraction of random configs collide in
                # this scene. Target: sparse ~10-15%, dense ~35-50%.
                reset_collision_count()
                col_rate = measure_collision_rate(n_samples=1000)
                print(f"    C-space collision rate: {col_rate*100:.1f}%")
                reset_collision_count()  # don't pollute planning metrics

                path, metrics = run_single_plan(
                    panda_id, obj_ids, table_id,
                    start, goal,
                )
                metrics["scene_attempts"] = scene_attempts
                metrics["collision_rate"] = round(col_rate * 100, 1)

            metrics["density"] = density
            metrics["num_objects"] = num_objects
            metrics["run_id"] = run_id

            all_runs.append(metrics)

            if metrics["success"]:
                print(f"    OK — {metrics['num_waypoints']} waypoints, "
                      f"{metrics['planning_time_s']:.2f}s, "
                      f"{metrics['collision_checks']} checks")
            else:
                print(f"    FAIL — {metrics.get('reason', 'no path found')}")

            if not args.headless and path:
                visualize_path(panda_id, path, arm_indices)
                time.sleep(1.0)

    print_summary(all_runs)

    if args.log:
        save_log(all_runs, args.log_dir)

    if args.plot:
        generate_plots(all_runs, args.fig_dir)

    p.disconnect()


if __name__ == "__main__":
    main()