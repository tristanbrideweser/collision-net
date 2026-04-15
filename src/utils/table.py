# src/utils/table.py

import json
import os
import numpy as np


def load_log(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def print_summary_table(runs):
    """Print a formatted summary table grouped by density."""
    densities = sorted(set(r["density"] for r in runs))

    header = (
        f"{'Density':<14} "
        f"{'Runs':>5} "
        f"{'Success%':>9} "
        f"{'Avg Time(s)':>12} "
        f"{'Avg Path Len':>13} "
        f"{'Avg Waypts':>11} "
        f"{'Avg Col Chks':>13} "
        f"{'Med Col Chks':>13}"
    )
    print("=" * len(header))
    print("RRT-Connect Planning Results (Geometric Collision Checker)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for density in densities:
        d_runs = [r for r in runs if r["density"] == density]
        total = len(d_runs)
        successes = [r for r in d_runs if r["success"]]
        success_rate = len(successes) / total * 100

        if successes:
            avg_time = np.mean([r["planning_time_s"] for r in successes])
            std_time = np.std([r["planning_time_s"] for r in successes])
            avg_path = np.mean([r["path_length"] for r in successes])
            avg_waypoints = np.mean([r["num_waypoints"] for r in successes])
            avg_checks = np.mean([r["collision_checks"] for r in successes])
            med_checks = np.median([r["collision_checks"] for r in successes])
        else:
            avg_time = std_time = avg_path = avg_waypoints = 0
            avg_checks = med_checks = 0

        # include failed runs in collision check average too
        avg_checks_all = np.mean([r["collision_checks"] for r in d_runs])

        print(
            f"{density:<14} "
            f"{total:>5} "
            f"{success_rate:>8.1f}% "
            f"{avg_time:>7.2f}±{std_time:<4.2f} "
            f"{avg_path:>13.2f} "
            f"{avg_waypoints:>11.1f} "
            f"{avg_checks_all:>13.0f} "
            f"{med_checks:>13.0f}"
        )

    print("=" * len(header))


def print_per_run_table(runs):
    """Print detailed per-run results."""
    header = (
        f"{'Density':<12} "
        f"{'Run':>4} "
        f"{'Status':>8} "
        f"{'Time(s)':>8} "
        f"{'Path Len':>9} "
        f"{'Waypts':>7} "
        f"{'Col Chks':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in runs:
        status = "OK" if r["success"] else "FAIL"
        print(
            f"{r['density']:<12} "
            f"{r['run_id']:>4} "
            f"{status:>8} "
            f"{r['planning_time_s']:>8.3f} "
            f"{r['path_length']:>9.3f} "
            f"{r['num_waypoints']:>7} "
            f"{r['collision_checks']:>9}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="logs/planning_log.json")
    parser.add_argument("--detail", action="store_true", help="Show per-run detail")
    args = parser.parse_args()

    runs = load_log(args.log)
    print_summary_table(runs)
    if args.detail:
        print()
        print_per_run_table(runs)