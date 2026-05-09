# src/scripts/plot_results.py
"""
Generate Milestone 1 figures from planning logs.

Usage:
    python plot_results.py --log logs/planning_log.json --out figures/
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_log(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def group_by_density(runs):
    groups = {}
    for r in runs:
        d = r["density"]
        if d not in groups:
            groups[d] = []
        groups[d].append(r)
    return groups


def plot_planning_time(groups, out_dir):
    """Bar chart: average planning time per density."""
    densities = ["sparse", "medium", "dense"]
    means, stds = [], []

    for d in densities:
        if d in groups:
            successes = [r["planning_time_s"] for r in groups[d] if r["success"]]
            means.append(np.mean(successes) if successes else 0)
            stds.append(np.std(successes) if successes else 0)
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(densities, means, yerr=stds, capsize=5,
                  color=["#4CAF50", "#FF9800", "#F44336"], alpha=0.85)
    ax.set_ylabel("Planning Time (s)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("RRT-Connect Planning Time vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "planning_time.png"), dpi=150)
    plt.close()
    print("Saved planning_time.png")


def plot_success_rate(groups, out_dir):
    """Bar chart: success rate per density."""
    densities = ["sparse", "medium", "dense"]
    rates = []

    for d in densities:
        if d in groups:
            total = len(groups[d])
            success = sum(1 for r in groups[d] if r["success"])
            rates.append(success / total * 100)
        else:
            rates.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(densities, rates,
           color=["#4CAF50", "#FF9800", "#F44336"], alpha=0.85)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Planning Success Rate vs Clutter Density")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_rate.png"), dpi=150)
    plt.close()
    print("Saved success_rate.png")


def plot_collision_checks(groups, out_dir):
    """Box plot: collision checks per density."""
    densities = ["sparse", "medium", "dense"]
    data = []

    for d in densities:
        if d in groups:
            checks = [r["collision_checks"] for r in groups[d]]
            data.append(checks)
        else:
            data.append([0])

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, labels=densities, patch_artist=True)
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Number of Collision Checks")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Collision Checks per Plan vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collision_checks.png"), dpi=150)
    plt.close()
    print("Saved collision_checks.png")


def plot_path_length(groups, out_dir):
    """Bar chart: average path length per density."""
    densities = ["sparse", "medium", "dense"]
    means, stds = [], []

    for d in densities:
        if d in groups:
            lengths = [r["path_length"] for r in groups[d] if r["success"]]
            means.append(np.mean(lengths) if lengths else 0)
            stds.append(np.std(lengths) if lengths else 0)
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(densities, means, yerr=stds, capsize=5,
           color=["#4CAF50", "#FF9800", "#F44336"], alpha=0.85)
    ax.set_ylabel("Path Length (joint-space)")
    ax.set_xlabel("Clutter Density")
    ax.set_title("Path Length vs Clutter Density")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "path_length.png"), dpi=150)
    plt.close()
    print("Saved path_length.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="logs/planning_log.json")
    parser.add_argument("--out", type=str, default="figures/")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_log(args.log)
    groups = group_by_density(runs)

    plot_planning_time(groups, args.out)
    plot_success_rate(groups, args.out)
    plot_collision_checks(groups, args.out)
    plot_path_length(groups, args.out)

    print(f"\nAll figures saved to {args.out}")


if __name__ == "__main__":
    main()