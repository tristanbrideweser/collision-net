"""
visualize_scenes.py — Visualize MPiNets environments in PyBullet GUI.

Usage:
    cd src/data/pipeline
    
    # View all scene types (press Enter to advance)
    python visualize_scenes.py ../envs/hybrid_solvable_problems.pkl
    
    # View specific scene types
    python visualize_scenes.py ../envs/hybrid_solvable_problems.pkl --scene-types tabletop cubby
    
    # View N scenes per type
    python visualize_scenes.py ../envs/hybrid_solvable_problems.pkl --max-scenes 3
    
    # Auto-advance every 2 seconds
    python visualize_scenes.py ../envs/hybrid_solvable_problems.pkl --auto 2.0
"""

# ──────────────────────────────────────────────
# Fake module registration (same as generate_dataset.py)
# ──────────────────────────────────────────────

import sys
import types

for _k in list(sys.modules.keys()):
    if "mpinets" in _k or "geometrout" in _k or "robofin" in _k:
        del sys.modules[_k]

_fake_mods = [
    "mpinets", "mpinets.utils", "mpinets.mpinets_types",
    "geometrout", "geometrout.primitive", "geometrout.transform",
    "robofin", "robofin.robots",
]
for _name in _fake_mods:
    sys.modules[_name] = types.ModuleType(_name)

import argparse
import os
import pickle
import time
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass, field
from typing import Optional, Union


# ──────────────────────────────────────────────
# MPiNets type shims
# ──────────────────────────────────────────────

@dataclass
class SE3:
    _xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _so3: object = None

@dataclass
class SO3:
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass
class Cuboid:
    _pose: Optional[SE3] = None
    _dims: np.ndarray = field(default_factory=lambda: np.ones(3))

@dataclass
class Cylinder:
    _pose: Optional[SE3] = None
    radius: float = 0.05
    height: float = 0.1

@dataclass
class Sphere:
    _pose: Optional[SE3] = None
    radius: float = 0.05

@dataclass
class PlanningProblem:
    target: Optional[SE3] = None
    target_volume: Optional[Union[Cuboid, Cylinder]] = None
    q0: np.ndarray = field(default_factory=lambda: np.zeros(7))
    obstacles: Optional[list] = None
    obstacle_point_cloud: Optional[np.ndarray] = None
    target_negative_volumes: list = field(default_factory=list)

@dataclass
class FrankaRobot:
    pass

for cls, mod in [
    (SE3, "geometrout.transform"), (SO3, "geometrout.transform"),
    (Cuboid, "geometrout.primitive"), (Cylinder, "geometrout.primitive"),
    (Sphere, "geometrout.primitive"), (PlanningProblem, "mpinets.mpinets_types"),
    (FrankaRobot, "robofin.robots"),
]:
    cls.__module__ = mod
    sys.modules[mod].__dict__[cls.__name__] = cls

sys.modules["mpinets.utils"].PlanningProblem = PlanningProblem


# ──────────────────────────────────────────────
# Scene spawning utilities
# ──────────────────────────────────────────────

def rotation_matrix_to_quaternion(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def get_obstacle_pose(obs):
    pose = getattr(obs, '_pose', None)
    if pose is None:
        return [0, 0, 0], [0, 0, 0, 1]
    xyz = getattr(pose, '_xyz', None)
    pos = list(xyz) if xyz is not None else [0, 0, 0]
    so3 = getattr(pose, '_so3', None)
    if so3 is not None:
        mat = getattr(so3, 'matrix', None)
        if mat is not None and hasattr(mat, 'shape') and mat.shape == (3, 3):
            orn = rotation_matrix_to_quaternion(mat)
        else:
            orn = [0, 0, 0, 1]
    else:
        orn = [0, 0, 0, 1]
    return pos, orn


# Color scheme per obstacle type
COLORS = {
    "Cuboid": [0.4, 0.6, 0.8, 0.9],    # blue-ish
    "Cylinder": [0.8, 0.5, 0.3, 0.9],  # orange-ish
    "Sphere": [0.6, 0.8, 0.4, 0.9],    # green-ish
}


def spawn_scene(obstacles):
    """Spawn obstacles with color-coded shapes."""
    obj_ids = []
    for obs in obstacles:
        pos, orn = get_obstacle_pose(obs)
        t = type(obs).__name__
        color = COLORS.get(t, [0.5, 0.5, 0.5, 0.8])

        if t == "Cuboid":
            dims = getattr(obs, '_dims', np.ones(3))
            half = [d / 2 for d in dims]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        elif t == "Cylinder":
            r = getattr(obs, 'radius', 0.05)
            h = getattr(obs, 'height', 0.1)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
        elif t == "Sphere":
            r = getattr(obs, 'radius', 0.05)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
        else:
            continue

        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=pos,
                                     baseOrientation=orn)
        obj_ids.append(body_id)
    return obj_ids


def clear_scene(obj_ids):
    for obj_id in obj_ids:
        p.removeBody(obj_id)


def set_robot_config(panda_id, config):
    """Set robot to a configuration."""
    for i in range(7):
        p.resetJointState(panda_id, i, config[i])


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize MPiNets environments")
    parser.add_argument("pkl_file", help="Path to MPiNets .pkl file")
    parser.add_argument("--scene-types", nargs="+",
                        default=["tabletop", "cubby", "merged_cubby", "dresser"])
    parser.add_argument("--categories", nargs="+",
                        default=["task_oriented", "neutral_start", "neutral_goal"])
    parser.add_argument("--max-scenes", type=int, default=5,
                        help="Max scenes to show per scene_type/category")
    parser.add_argument("--auto", type=float, default=None,
                        help="Auto-advance interval in seconds (default: manual with Enter)")
    parser.add_argument("--show-q0", action="store_true",
                        help="Set robot to initial config q0 from problem")
    args = parser.parse_args()

    # Load problems
    print(f"Loading {args.pkl_file}...")
    with open(args.pkl_file, "rb") as f:
        all_problems = pickle.load(f)

    # Collect scenes
    scene_list = []
    for scene_type in args.scene_types:
        if scene_type not in all_problems:
            print(f"  Skipping {scene_type} (not in pickle)")
            continue
        categories = all_problems[scene_type]
        for cat_name in args.categories:
            if cat_name not in categories:
                continue
            probs = categories[cat_name]
            if not isinstance(probs, list):
                continue
            limit = args.max_scenes if args.max_scenes else len(probs)
            for i in range(min(limit, len(probs))):
                scene_list.append((scene_type, cat_name, probs[i]))

    print(f"Found {len(scene_list)} scenes to visualize")
    if len(scene_list) == 0:
        return

    # Setup PyBullet GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setGravity(0, 0, -9.81)
    
    # Set camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0, 0.4]
    )

    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Visualization loop
    obj_ids = []
    print("\n" + "="*60)
    print("CONTROLS:")
    if args.auto:
        print(f"  Auto-advancing every {args.auto:.1f} seconds")
    else:
        print("  Press ENTER to advance to next scene")
    print("  Press 'q' + ENTER to quit")
    print("="*60 + "\n")

    for idx, (scene_type, cat_name, prob) in enumerate(scene_list):
        # Clear previous
        if obj_ids:
            clear_scene(obj_ids)

        # Spawn new scene
        obj_ids = spawn_scene(prob.obstacles)
        
        # Set robot config
        if args.show_q0 and prob.q0 is not None:
            set_robot_config(panda_id, prob.q0)
        else:
            # Default pose
            set_robot_config(panda_id, [0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        # Count obstacle types
        type_counts = {}
        for obs in prob.obstacles:
            t = type(obs).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
        type_str = ", ".join(f"{v} {k}" for k, v in type_counts.items())

        print(f"[{idx+1}/{len(scene_list)}] {scene_type} / {cat_name}")
        print(f"         Obstacles: {len(prob.obstacles)} ({type_str})")

        # Wait for input or auto-advance
        if args.auto:
            time.sleep(args.auto)
        else:
            user_input = input("         > Press Enter (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break

    p.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()