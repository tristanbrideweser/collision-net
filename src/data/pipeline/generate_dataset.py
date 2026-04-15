"""
generate_dataset.py — Generate collision detection training data from MPiNets scenes.

Loads MPiNets problem pickle files, reconstructs each scene in PyBullet,
samples random configs, labels them with the geometric collision checker,
and outputs training data for PointNet.

Directory structure:
    src/
    ├── data/
    │   ├── envs/                <- MPiNets .pkl files
    │   │   └── hybrid_solvable_problems.pkl
    │   ├── pipeline/            <- this script
    │   │   └── generate_dataset.py
    │   └── scenes/              <- generated output
    │       ├── scene_0000.npz
    │       ├── train.npz
    │       ├── val.npz
    │       └── test.npz
    ├── planner/
    │   └── collision_detector.py
    └── model/

Usage:
    cd src/data/pipeline
    python generate_dataset.py ../envs/hybrid_solvable_problems.pkl
    python generate_dataset.py ../envs/hybrid_solvable_problems.pkl --scene-types tabletop cubby
    python generate_dataset.py ../envs/hybrid_solvable_problems.pkl --samples-per-scene 5000 --max-scenes 50
"""

# ──────────────────────────────────────────────
# Register fake modules FIRST so pickle can resolve
# MPiNets types without their full stack installed.
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

# Now safe to import everything else
import argparse
import os
import random
import pickle
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass, field
from typing import Optional, Union

# Add src/ to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from planner.collision_detector import (
    init_collision_checker,
    update_obstacles,
    in_collision,
    JOINT_LIMITS,
)

# Derived from JOINT_LIMITS
JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])
JOINT_RANGE = JOINT_UPPER - JOINT_LOWER


# ──────────────────────────────────────────────
# MPiNets type shims (for unpickling)
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

# Wire into fake modules
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
# PyBullet scene reconstruction from MPiNets obstacles
# ──────────────────────────────────────────────

def get_obstacle_pose(obs):
    """Extract position and orientation from an MPiNets obstacle."""
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


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to [x, y, z, w] quaternion (PyBullet convention)."""
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


def spawn_mpinets_scene(obstacles):
    """Reconstruct an MPiNets scene in PyBullet."""
    obj_ids = []

    for obs in obstacles:
        pos, orn = get_obstacle_pose(obs)
        t = type(obs).__name__

        if t == "Cuboid":
            dims = getattr(obs, '_dims', np.ones(3))
            half_extents = [d / 2 for d in dims]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                      rgbaColor=[0.6, 0.6, 0.6, 0.8])
        elif t == "Cylinder":
            radius = getattr(obs, 'radius', 0.05)
            height = getattr(obs, 'height', 0.1)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                      rgbaColor=[0.5, 0.5, 0.7, 0.8])
        elif t == "Sphere":
            radius = getattr(obs, 'radius', 0.05)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                      rgbaColor=[0.7, 0.5, 0.5, 0.8])
        else:
            continue

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=orn,
        )
        obj_ids.append(body_id)

    return obj_ids


def clear_scene(obj_ids):
    """Remove all obstacle bodies from PyBullet."""
    for obj_id in obj_ids:
        p.removeBody(obj_id)


# ──────────────────────────────────────────────
# Point cloud extraction
# ──────────────────────────────────────────────

def _sample_box_surface(half_extents, n=200):
    hx, hy, hz = half_extents
    faces, area = [], []
    for sign in [-1, 1]:
        faces.append(lambda n, s=sign, hy=hy, hz=hz, hx=hx: np.column_stack([
            np.full(n, s * hx), np.random.uniform(-hy, hy, n), np.random.uniform(-hz, hz, n)]))
        area.append(4 * hy * hz)
    for sign in [-1, 1]:
        faces.append(lambda n, s=sign, hx=hx, hz=hz, hy=hy: np.column_stack([
            np.random.uniform(-hx, hx, n), np.full(n, s * hy), np.random.uniform(-hz, hz, n)]))
        area.append(4 * hx * hz)
    for sign in [-1, 1]:
        faces.append(lambda n, s=sign, hx=hx, hy=hy, hz=hz: np.column_stack([
            np.random.uniform(-hx, hx, n), np.random.uniform(-hy, hy, n), np.full(n, s * hz)]))
        area.append(4 * hx * hy)
    total_area = sum(area)
    points = []
    for face_fn, a in zip(faces, area):
        face_n = max(1, int(n * a / total_area))
        points.append(face_fn(face_n))
    return np.concatenate(points, axis=0)


def _sample_cylinder_surface(radius, height, n=200):
    side_area = 2 * np.pi * radius * height
    cap_area = 2 * np.pi * radius ** 2
    total = side_area + cap_area
    n_side = max(1, int(n * side_area / total))
    n_caps = n - n_side
    theta = np.random.uniform(0, 2 * np.pi, n_side)
    z = np.random.uniform(-height / 2, height / 2, n_side)
    side = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    cap_points = []
    for cap_z in [-height / 2, height / 2]:
        nc = max(1, n_caps // 2)
        r = radius * np.sqrt(np.random.uniform(0, 1, nc))
        theta = np.random.uniform(0, 2 * np.pi, nc)
        cap_points.append(np.column_stack([
            r * np.cos(theta), r * np.sin(theta), np.full(nc, cap_z)]))
    return np.concatenate([side] + cap_points, axis=0)


def _sample_sphere_surface(radius, n=200):
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return np.column_stack([
        radius * sin_theta * np.cos(phi),
        radius * sin_theta * np.sin(phi),
        radius * cos_theta])


def extract_scene_point_cloud(obstacle_ids, n_points=2048, points_per_shape=300):
    """Sample surface points from all obstacles via PyBullet collision shapes."""
    all_points = []
    for obj_id in obstacle_ids:
        shape_data = p.getCollisionShapeData(obj_id, -1)
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        for shape in shape_data:
            geom_type = shape[2]
            dims = shape[3]
            local_pos = shape[5]
            local_orn = shape[6]
            if geom_type == p.GEOM_BOX:
                pts = _sample_box_surface(dims, n=points_per_shape)
            elif geom_type == p.GEOM_CYLINDER:
                pts = _sample_cylinder_surface(dims[1], dims[0], n=points_per_shape)
            elif geom_type == p.GEOM_SPHERE:
                pts = _sample_sphere_surface(dims[0], n=points_per_shape)
            else:
                continue
            # Transform to world frame
            local_mat = np.array(p.getMatrixFromQuaternion(local_orn)).reshape(3, 3)
            pts = (local_mat @ pts.T).T + np.array(local_pos)
            body_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            pts = (body_mat @ pts.T).T + np.array(pos)
            all_points.append(pts)

    if len(all_points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    all_points = np.concatenate(all_points, axis=0)
    if len(all_points) >= n_points:
        indices = np.random.choice(len(all_points), n_points, replace=False)
    else:
        indices = np.random.choice(len(all_points), n_points, replace=True)
    return all_points[indices].astype(np.float32)


# ──────────────────────────────────────────────
# Config sampling utilities
# ──────────────────────────────────────────────

def sample_random_config():
    """Sample a random configuration within joint limits."""
    return np.random.uniform(JOINT_LOWER, JOINT_UPPER)


def normalize_config(config):
    """Normalize config to [-1, 1] range."""
    return 2.0 * (np.array(config) - JOINT_LOWER) / JOINT_RANGE - 1.0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def get_args():
    ap = argparse.ArgumentParser(description="Generate training data from MPiNets scenes")
    ap.add_argument("pkl_file", help="Path to MPiNets .pkl file (e.g., ../envs/hybrid_solvable_problems.pkl)")
    ap.add_argument("--scene-types", nargs="+",
                    default=["tabletop", "cubby", "merged_cubby", "dresser"],
                    help="Scene types to use")
    ap.add_argument("--categories", nargs="+",
                    default=["task_oriented", "neutral_start", "neutral_goal"],
                    help="Problem categories to use")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="Max scenes per scene_type/category (None = all)")
    ap.add_argument("--samples-per-scene", type=int, default=2000)
    ap.add_argument("--n-points", type=int, default=2048,
                    help="Number of points in obstacle point cloud")
    ap.add_argument("--output-dir", type=str, default="../scenes",
                    help="Output directory (default: ../scenes relative to pipeline/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    return ap.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve output directory relative to script location
    output_dir = os.path.normpath(os.path.join(SCRIPT_DIR, args.output_dir))
    
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Output directory: {output_dir}")

    # Load MPiNets problems
    print(f"Loading {args.pkl_file}...")
    with open(args.pkl_file, "rb") as f:
        all_problems = pickle.load(f)

    # Setup PyBullet
    if args.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Collect all scenes to process
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

    print(f"Total scenes to process: {len(scene_list)}")

    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    scenes_subdir = os.path.join(output_dir, "scenes")
    os.makedirs(scenes_subdir, exist_ok=True)

    all_scene_ids = []
    all_configs = []
    all_labels = []
    obj_ids = []

    for scene_idx, (scene_type, cat_name, prob) in enumerate(scene_list):
        # Clear previous obstacles
        if scene_idx > 0:
            clear_scene(obj_ids)

        # Spawn MPiNets obstacles in PyBullet
        obj_ids = spawn_mpinets_scene(prob.obstacles)

        # Extract point cloud
        point_cloud = extract_scene_point_cloud(obj_ids, n_points=args.n_points)

        # Save scene
        scene_file = os.path.join(scenes_subdir, f"scene_{scene_idx:04d}.npz")
        np.savez_compressed(scene_file,
                            point_cloud=point_cloud,
                            scene_type=scene_type,
                            category=cat_name,
                            q0=prob.q0)

        # Initialize collision checker with current scene
        init_collision_checker(panda_id=panda_id, obstacle_ids=obj_ids, table_id=None)

        # Sample configs and label
        n_collision = 0
        for _ in range(args.samples_per_scene):
            config = sample_random_config()
            label = int(in_collision(config.tolist()))
            n_collision += label

            all_scene_ids.append(scene_idx)
            all_configs.append(normalize_config(config))
            all_labels.append(label)

        col_rate = n_collision / args.samples_per_scene * 100
        print(f"  Scene {scene_idx:4d} ({scene_type}/{cat_name}): "
              f"{len(prob.obstacles)} obstacles, "
              f"{n_collision}/{args.samples_per_scene} collisions ({col_rate:.1f}%)")

    # Convert to arrays
    all_scene_ids = np.array(all_scene_ids, dtype=np.int32)
    all_configs = np.array(all_configs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    total = len(all_labels)
    print(f"\nTotal samples: {total}")
    print(f"  Collision:  {all_labels.sum()} ({all_labels.mean()*100:.1f}%)")
    print(f"  Free:       {total - all_labels.sum()} ({(1-all_labels.mean())*100:.1f}%)")

    # Shuffle and split
    indices = np.random.permutation(total)
    train_end = int(total * args.train_ratio)
    val_end = int(total * (args.train_ratio + args.val_ratio))

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, split_idx in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(
            filepath,
            scene_ids=all_scene_ids[split_idx],
            configs=all_configs[split_idx],
            labels=all_labels[split_idx],
        )
        n_col = all_labels[split_idx].sum()
        n_total = len(split_idx)
        print(f"  {split_name}: {n_total} samples ({n_col} collisions, "
              f"{n_col/n_total*100:.1f}%) -> {filepath}")

    p.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()