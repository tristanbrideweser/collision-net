"""
demo_collision_checkers_v2.py — Visual comparison using real MPiNets obstacles.

Loads actual obstacle geometry from the MPiNets pickle file instead of
approximating with spheres from the point cloud.

Usage:
    cd src/planner
    python demo_collision_checkers_v2.py --model ../model/checkpoints/best.pt --pkl ../data/envs/hybrid_solvable_problems.pkl
    
    # Specific scene type
    python demo_collision_checkers_v2.py --model ../model/checkpoints/best.pt --pkl ../data/envs/hybrid_solvable_problems.pkl --scene-type tabletop
    
    # Slower animation
    python demo_collision_checkers_v2.py --model ../model/checkpoints/best.pt --pkl ../data/envs/hybrid_solvable_problems.pkl --delay 1.0
"""

# ──────────────────────────────────────────────
# Fake module registration for MPiNets pickle
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

# Add src/ to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


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


# Now import collision checkers
from planner.collision_detector import (
    init_collision_checker,
    in_collision as geometric_in_collision,
    JOINT_LIMITS,
)
from planner.nn_collision_detector import NeuralCollisionChecker

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])

# Colors
COLOR_BOTH_FREE = [0.2, 0.8, 0.2, 1.0]       # Green
COLOR_BOTH_COLLISION = [0.8, 0.2, 0.2, 1.0]  # Red
COLOR_FALSE_POSITIVE = [1.0, 0.8, 0.0, 1.0]  # Yellow
COLOR_FALSE_NEGATIVE = [0.6, 0.0, 0.8, 1.0]  # Purple

OBSTACLE_COLORS = {
    "Cuboid": [0.4, 0.5, 0.7, 0.85],
    "Cylinder": [0.7, 0.5, 0.4, 0.85],
    "Sphere": [0.5, 0.7, 0.4, 0.85],
}


# ──────────────────────────────────────────────
# Obstacle spawning (real geometry)
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


def spawn_mpinets_obstacles(obstacles):
    """Spawn real MPiNets obstacles in PyBullet."""
    obj_ids = []
    for obs in obstacles:
        pos, orn = get_obstacle_pose(obs)
        t = type(obs).__name__
        color = OBSTACLE_COLORS.get(t, [0.5, 0.5, 0.5, 0.8])

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


def clear_obstacles(obj_ids):
    for obj_id in obj_ids:
        p.removeBody(obj_id)


# ──────────────────────────────────────────────
# Point cloud extraction (for neural checker)
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


def extract_point_cloud(obstacle_ids, n_points=2048, points_per_shape=300):
    """Sample surface points from PyBullet obstacles."""
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
# Robot utilities
# ──────────────────────────────────────────────

def set_robot_color(panda_id, color):
    for link_id in range(-1, 11):
        p.changeVisualShape(panda_id, link_id, rgbaColor=color)


def set_robot_config(panda_id, config):
    for i in range(7):
        p.resetJointState(panda_id, i, config[i])


def sample_random_config():
    return np.random.uniform(JOINT_LOWER, JOINT_UPPER)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def run_demo(neural_checker, panda_id, n_configs=50, delay=0.5):
    stats = {"both_free": 0, "both_col": 0, "fp": 0, "fn": 0}
    
    for i in range(n_configs):
        config = sample_random_config()
        
        neural_pred = neural_checker.in_collision(config.tolist())
        neural_prob = neural_checker.collision_probability(config.tolist())
        geometric_pred = geometric_in_collision(config.tolist())
        
        if not neural_pred and not geometric_pred:
            color = COLOR_BOTH_FREE
            status = "FREE"
            stats["both_free"] += 1
        elif neural_pred and geometric_pred:
            color = COLOR_BOTH_COLLISION
            status = "COLLISION"
            stats["both_col"] += 1
        elif neural_pred and not geometric_pred:
            color = COLOR_FALSE_POSITIVE
            status = "FALSE POS"
            stats["fp"] += 1
        else:
            color = COLOR_FALSE_NEGATIVE
            status = "FALSE NEG ⚠️"
            stats["fn"] += 1
        
        set_robot_config(panda_id, config)
        set_robot_color(panda_id, color)
        
        print(f"[{i+1:3d}/{n_configs}] {status:12s} | Neural: {neural_prob:.2f} | "
              f"Geo: {'COL' if geometric_pred else 'FREE'}")
        
        time.sleep(delay)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Demo collision checkers with real MPiNets obstacles")
    parser.add_argument("--model", type=str, default="../model/checkpoints/best.pt")
    parser.add_argument("--pkl", type=str, default="../data/envs/hybrid_solvable_problems.pkl",
                        help="MPiNets pickle file")
    parser.add_argument("--scene-type", type=str, default=None,
                        choices=["tabletop", "cubby", "merged_cubby", "dresser"],
                        help="Specific scene type (default: all)")
    parser.add_argument("--category", type=str, default="task_oriented")
    parser.add_argument("--n-scenes", type=int, default=3)
    parser.add_argument("--n-configs", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load MPiNets problems
    print(f"Loading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        all_problems = pickle.load(f)
    
    # Collect scenes
    scene_types = [args.scene_type] if args.scene_type else ["tabletop", "cubby", "merged_cubby", "dresser"]
    scenes = []
    for st in scene_types:
        if st not in all_problems:
            continue
        if args.category in all_problems[st]:
            probs = all_problems[st][args.category]
            if isinstance(probs, list):
                for prob in probs[:args.n_scenes]:
                    scenes.append((st, prob))
    
    if not scenes:
        print("No scenes found!")
        return
    
    print(f"Found {len(scenes)} scenes")
    
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8, cameraYaw=45, cameraPitch=-25,
        cameraTargetPosition=[0.3, 0, 0.3]
    )
    
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    # Initialize neural checker
    dummy_pc = np.zeros((2048, 3), dtype=np.float32)
    neural_checker = NeuralCollisionChecker(
        scene_point_cloud=dummy_pc,
        model_path=args.model,
        threshold=args.threshold,
    )
    
    print("\n" + "="*60)
    print("COLOR LEGEND:")
    print("  🟢 GREEN  - Both agree: FREE")
    print("  🔴 RED    - Both agree: COLLISION")
    print("  🟡 YELLOW - False Positive (Neural=col, Geo=free)")
    print("  🟣 PURPLE - False Negative (Neural=free, Geo=col) ⚠️")
    print("="*60)
    print("\nPress Ctrl+C to exit\n")
    
    total_stats = {"both_free": 0, "both_col": 0, "fp": 0, "fn": 0}
    obstacle_ids = []
    
    try:
        for scene_idx, (scene_type, prob) in enumerate(scenes):
            if obstacle_ids:
                clear_obstacles(obstacle_ids)
            
            print(f"\n{'='*60}")
            print(f"Scene {scene_idx+1}/{len(scenes)}: {scene_type}")
            print(f"Obstacles: {len(prob.obstacles)}")
            print(f"{'='*60}")
            
            # Spawn real obstacles
            obstacle_ids = spawn_mpinets_obstacles(prob.obstacles)
            
            # Init geometric checker
            init_collision_checker(panda_id=panda_id, obstacle_ids=obstacle_ids, table_id=None)
            
            # Extract point cloud for neural checker
            point_cloud = extract_point_cloud(obstacle_ids, n_points=2048)
            neural_checker.update_scene(point_cloud)
            
            # Show initial config if available
            if prob.q0 is not None and len(prob.q0) == 7:
                set_robot_config(panda_id, prob.q0)
                time.sleep(0.5)
            
            # Run demo
            stats = run_demo(neural_checker, panda_id, args.n_configs, args.delay)
            
            for k in total_stats:
                total_stats[k] += stats[k]
            
            total = sum(stats.values())
            print(f"\nScene Summary:")
            print(f"  Agreement:       {stats['both_free'] + stats['both_col']:3d} ({(stats['both_free'] + stats['both_col'])/total*100:.1f}%)")
            print(f"  False Positives: {stats['fp']:3d}")
            print(f"  False Negatives: {stats['fn']:3d} {'⚠️' if stats['fn'] > 0 else '✓'}")
            
            if scene_idx < len(scenes) - 1:
                input("\nPress Enter for next scene...")
        
        # Final summary
        total = sum(total_stats.values())
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        accuracy = (total_stats['both_free'] + total_stats['both_col']) / total
        print(f"Accuracy: {accuracy*100:.1f}%")
        print(f"False Negatives: {total_stats['fn']} ({total_stats['fn']/total*100:.1f}%)")
        
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    p.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()