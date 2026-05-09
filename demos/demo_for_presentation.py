"""
demo_for_presentation.py — Full demo of neural collision detection for CS558 presentation.

This script:
  1. Loads a real MPiNets scene with obstacles
  2. Extracts point cloud for the neural checker
  3. Plans a path using RRT-Connect with neural collision checking
  4. Visualizes the robot executing the path in PyBullet

Usage:
    cd src/planner
    python demo_for_presentation.py \
        --model ../model/checkpoints/best.pt \
        --pkl ../data/envs/hybrid_solvable_problems.pkl \
        --scene-type tabletop
"""

# ══════════════════════════════════════════════════════════════════════════════
# MPiNets pickle shims (must come before other imports)
# ══════════════════════════════════════════════════════════════════════════════

import sys
import types
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

# Clear any cached mpinets modules
for _k in list(sys.modules.keys()):
    if "mpinets" in _k or "geometrout" in _k or "robofin" in _k:
        del sys.modules[_k]

# Create fake modules
for _name in ["mpinets", "mpinets.utils", "mpinets.mpinets_types",
              "geometrout", "geometrout.primitive", "geometrout.transform",
              "robofin", "robofin.robots"]:
    sys.modules[_name] = types.ModuleType(_name)

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

# Register classes in fake modules
for cls, mod in [(SE3, "geometrout.transform"), (SO3, "geometrout.transform"),
                 (Cuboid, "geometrout.primitive"), (Cylinder, "geometrout.primitive"),
                 (Sphere, "geometrout.primitive"), (PlanningProblem, "mpinets.mpinets_types"),
                 (FrankaRobot, "robofin.robots")]:
    cls.__module__ = mod
    sys.modules[mod].__dict__[cls.__name__] = cls
sys.modules["mpinets.utils"].PlanningProblem = PlanningProblem


# ══════════════════════════════════════════════════════════════════════════════
# Main imports
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import pickle
import random
import time
from pathlib import Path

import pybullet as p
import pybullet_data

# Add src/ to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collision_detector import JOINT_LIMITS, init_collision_checker, in_collision as geometric_collision
from nn_collision_detector import NeuralCollisionChecker


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])

# Predefined robot configurations
WAYPOINTS = {
    "home":       np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    "ready":      np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.785]),
    "left":       np.array([0.8, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785]),
    "right":      np.array([-0.8, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785]),
    "forward":    np.array([0.0, 0.2, 0.0, -1.5, 0.0, 1.7, 0.785]),
    "high":       np.array([0.0, -1.0, 0.0, -1.2, 0.0, 2.5, 0.785]),
}

# Colors
COLOR_FREE = [0.2, 0.8, 0.2, 1.0]      # Green
COLOR_PATH = [0.2, 0.5, 0.9, 1.0]      # Blue
COLOR_COLLISION = [0.9, 0.2, 0.2, 1.0] # Red
COLOR_GOAL = [0.9, 0.7, 0.2, 1.0]      # Gold


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle handling
# ══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (R[2,1] - R[1,2]) * s, (R[0,2] - R[2,0]) * s, (R[1,0] - R[0,1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2,1] - R[1,2]) / s, 0.25 * s, (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s, 0.25 * s, (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s, (R[1,2] + R[2,1]) / s, 0.25 * s
    return [x, y, z, w]


def get_obstacle_pose(obs):
    """Extract position and orientation from MPiNets obstacle."""
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


def spawn_obstacles(obstacles):
    """Spawn MPiNets obstacles in PyBullet, return body IDs."""
    colors = {"Cuboid": [0.4, 0.5, 0.7, 0.85], "Cylinder": [0.7, 0.5, 0.4, 0.85], "Sphere": [0.5, 0.7, 0.4, 0.85]}
    obj_ids = []
    
    for obs in obstacles:
        pos, orn = get_obstacle_pose(obs)
        t = type(obs).__name__
        color = colors.get(t, [0.5, 0.5, 0.5, 0.8])

        if t == "Cuboid":
            dims = getattr(obs, '_dims', np.ones(3))
            half = [float(d) / 2 for d in dims]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        elif t == "Cylinder":
            r, h = getattr(obs, 'radius', 0.05), getattr(obs, 'height', 0.1)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
        elif t == "Sphere":
            r = getattr(obs, 'radius', 0.05)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
        else:
            continue

        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=orn)
        obj_ids.append(body_id)
    
    return obj_ids


# ══════════════════════════════════════════════════════════════════════════════
# Point cloud extraction
# ══════════════════════════════════════════════════════════════════════════════

def sample_box_surface(half_extents, n=200):
    hx, hy, hz = half_extents
    faces = [
        lambda n, hx=hx, hy=hy, hz=hz: np.column_stack([np.full(n, hx), np.random.uniform(-hy, hy, n), np.random.uniform(-hz, hz, n)]),
        lambda n, hx=hx, hy=hy, hz=hz: np.column_stack([np.full(n, -hx), np.random.uniform(-hy, hy, n), np.random.uniform(-hz, hz, n)]),
        lambda n, hx=hx, hy=hy, hz=hz: np.column_stack([np.random.uniform(-hx, hx, n), np.full(n, hy), np.random.uniform(-hz, hz, n)]),
        lambda n, hx=hx, hy=hy, hz=hz: np.column_stack([np.random.uniform(-hx, hx, n), np.full(n, -hy), np.random.uniform(-hz, hz, n)]),
        lambda n, hx=hx, hy=hy, hz=hz: np.column_stack([np.random.uniform(-hx, hx, n), np.random.uniform(-hy, hy, n), np.full(n, hz)]),
        lambda n, hx=hx, hy=hy, hz=hz: np.column_stack([np.random.uniform(-hx, hx, n), np.random.uniform(-hy, hy, n), np.full(n, -hz)]),
    ]
    pts = np.concatenate([f(n // 6 + 1) for f in faces], axis=0)
    return pts[:n]


def sample_cylinder_surface(radius, height, n=200):
    n_side = int(n * 0.7)
    theta = np.random.uniform(0, 2 * np.pi, n_side)
    z = np.random.uniform(-height / 2, height / 2, n_side)
    side = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    
    n_caps = n - n_side
    caps = []
    for cap_z in [-height / 2, height / 2]:
        nc = n_caps // 2
        r = radius * np.sqrt(np.random.uniform(0, 1, nc))
        theta = np.random.uniform(0, 2 * np.pi, nc)
        caps.append(np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(nc, cap_z)]))
    
    return np.concatenate([side] + caps, axis=0)


def sample_sphere_surface(radius, n=200):
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return np.column_stack([radius * sin_theta * np.cos(phi), radius * sin_theta * np.sin(phi), radius * cos_theta])


def extract_point_cloud(obstacle_ids, n_points=2048):
    """Extract point cloud from PyBullet obstacle bodies."""
    all_points = []
    
    for obj_id in obstacle_ids:
        shape_data = p.getCollisionShapeData(obj_id, -1)
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        
        for shape in shape_data:
            geom_type, dims = shape[2], shape[3]
            
            if geom_type == p.GEOM_BOX:
                pts = sample_box_surface(dims, n=300)
            elif geom_type == p.GEOM_CYLINDER:
                pts = sample_cylinder_surface(dims[1], dims[0], n=300)
            elif geom_type == p.GEOM_SPHERE:
                pts = sample_sphere_surface(dims[0], n=300)
            else:
                continue
            
            # Transform to world frame
            body_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            pts = (body_mat @ pts.T).T + np.array(pos)
            all_points.append(pts)

    if not all_points:
        return np.zeros((n_points, 3), dtype=np.float32)
    
    all_points = np.concatenate(all_points, axis=0)
    idx = np.random.choice(len(all_points), n_points, replace=len(all_points) < n_points)
    return all_points[idx].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Robot utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_robot_config(panda_id, config):
    for i in range(7):
        p.resetJointState(panda_id, i, config[i])


def set_robot_color(panda_id, color):
    for link_id in range(-1, 11):
        p.changeVisualShape(panda_id, link_id, rgbaColor=color)


# ══════════════════════════════════════════════════════════════════════════════
# RRT-Connect with Neural Collision Checking
# ══════════════════════════════════════════════════════════════════════════════

class RRTNode:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None


def rrt_connect(start, goal, checker, step_size=0.15, max_iter=3000):
    """RRT-Connect using neural collision checker."""
    
    def sample_random():
        return np.array([random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
    
    def nearest(q, tree):
        dists = [np.linalg.norm(q - n.conf) for n in tree]
        return tree[np.argmin(dists)]
    
    def steer(q_from, q_to):
        direction = q_to - q_from
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return None
        if dist <= step_size:
            return q_to.copy()
        return q_from + direction * (step_size / dist)
    
    def edge_free(q_from, q_to):
        dist = np.linalg.norm(q_to - q_from)
        n_steps = max(2, int(np.ceil(dist / (step_size / 2))))
        waypoints = np.linspace(q_from, q_to, n_steps)
        return not checker.batch_in_collision(waypoints).any()
    
    def connect(q_from, q_to):
        """Greedily extend toward target, return farthest reachable point."""
        dist = np.linalg.norm(q_to - q_from)
        n_steps = max(2, int(np.ceil(dist / (step_size / 2))))
        waypoints = np.linspace(q_from, q_to, n_steps)
        collisions = checker.batch_in_collision(waypoints)
        if not collisions.any():
            return q_to
        first_col = collisions.argmax()
        if first_col == 0:
            return None
        return waypoints[first_col - 1]
    
    # Check start/goal
    if checker.in_collision(start):
        print("  ERROR: Start in collision!")
        return None
    if checker.in_collision(goal):
        print("  ERROR: Goal in collision!")
        return None
    
    tree_a = [RRTNode(start)]
    tree_b = [RRTNode(goal)]
    swapped = False
    
    for i in range(max_iter):
        # Sample (with 5% goal bias)
        q_rand = tree_b[0].conf if random.random() < 0.05 else sample_random()
        
        # Extend tree_a
        q_near_a = nearest(q_rand, tree_a)
        q_new = steer(q_near_a.conf, q_rand)
        
        if q_new is not None and edge_free(q_near_a.conf, q_new):
            node_new = RRTNode(q_new)
            node_new.parent = q_near_a
            tree_a.append(node_new)
            
            # Try to connect tree_b
            q_near_b = nearest(q_new, tree_b)
            q_connect = connect(q_near_b.conf, q_new)
            
            if q_connect is not None:
                node_connect = RRTNode(q_connect)
                node_connect.parent = q_near_b
                tree_b.append(node_connect)
                
                if np.linalg.norm(q_connect - q_new) < 1e-3:
                    # Connected! Extract path
                    path_a = []
                    curr = node_new
                    while curr:
                        path_a.append(curr.conf)
                        curr = curr.parent
                    path_a.reverse()
                    
                    path_b = []
                    curr = node_connect
                    while curr:
                        path_b.append(curr.conf)
                        curr = curr.parent
                    
                    path = path_b[::-1] + path_a if swapped else path_a + path_b
                    print(f"  Found path in {i+1} iterations ({len(path)} waypoints)")
                    return path
        
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped
        
        if (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}/{max_iter}...")
    
    print(f"  Failed after {max_iter} iterations")
    return None


def interpolate_path(path, max_step=0.03):
    """Interpolate path for smooth visualization."""
    result = [path[0]]
    for i in range(1, len(path)):
        dist = np.linalg.norm(path[i] - path[i-1])
        n = max(2, int(np.ceil(dist / max_step)))
        for j in range(1, n + 1):
            t = j / n
            result.append((1 - t) * path[i-1] + t * path[i])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Neural Collision Detection Demo")
    parser.add_argument("--model", default="../model/checkpoints/best.pt")
    parser.add_argument("--pkl", default="../data/envs/hybrid_solvable_problems.pkl")
    parser.add_argument("--scene-type", default="tabletop", choices=["tabletop", "cubby", "merged_cubby", "dresser"])
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # ── Load scene ──
    print(f"\n{'='*60}")
    print("LOADING SCENE")
    print('='*60)
    print(f"Loading {args.pkl}...")
    
    with open(args.pkl, "rb") as f:
        all_problems = pickle.load(f)
    
    # Find scene
    prob = None
    for cat in ["task_oriented", "neutral_start", "neutral_goal"]:
        if args.scene_type in all_problems and cat in all_problems[args.scene_type]:
            probs = all_problems[args.scene_type][cat]
            if isinstance(probs, list) and args.scene_index < len(probs):
                prob = probs[args.scene_index]
                break
    
    if prob is None:
        print("Scene not found!")
        return
    
    print(f"Scene: {args.scene_type}, {len(prob.obstacles)} obstacles")
    
    # ── Setup PyBullet ──
    print(f"\n{'='*60}")
    print("SETTING UP PYBULLET")
    print('='*60)
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=45, cameraPitch=-25, cameraTargetPosition=[0.3, 0, 0.4])
    
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    # Spawn obstacles
    obstacle_ids = spawn_obstacles(prob.obstacles)
    print(f"Spawned {len(obstacle_ids)} obstacles")
    
    # ── Extract point cloud ──
    print(f"\n{'='*60}")
    print("EXTRACTING POINT CLOUD")
    print('='*60)
    
    point_cloud = extract_point_cloud(obstacle_ids, n_points=2048)
    print(f"Point cloud shape: {point_cloud.shape}")
    
    # ── Initialize checkers ──
    print(f"\n{'='*60}")
    print("INITIALIZING COLLISION CHECKERS")
    print('='*60)
    
    neural_checker = NeuralCollisionChecker(
        scene_point_cloud=point_cloud,
        panda_id=panda_id,
        model_path=args.model,
        threshold=args.threshold,
    )
    
    init_collision_checker(panda_id=panda_id, obstacle_ids=obstacle_ids, table_id=None)
    
    # ── Define start and goal ──
    start = WAYPOINTS["home"]
    
    # Try to find a collision-free goal
    print("\nSampling collision-free goal...")
    goal = None
    for name, wp in WAYPOINTS.items():
        if name != "home" and not neural_checker.in_collision(wp):
            goal = wp
            print(f"  Using '{name}' as goal")
            break
    
    if goal is None:
        # Sample random goal
        for _ in range(500):
            candidate = np.array([random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
            if not neural_checker.in_collision(candidate):
                goal = candidate
                print("  Using random goal")
                break
    
    if goal is None:
        print("Could not find collision-free goal!")
        input("Press Enter to exit...")
        p.disconnect()
        return
    
    # ── Show start ──
    set_robot_config(panda_id, start)
    set_robot_color(panda_id, COLOR_FREE)
    print(f"\nStart: {np.round(start, 2)}")
    print(f"Goal:  {np.round(goal, 2)}")
    
    input("\n>>> Press Enter to start planning...")
    
    # ── Plan ──
    print(f"\n{'='*60}")
    print("PLANNING WITH NEURAL COLLISION CHECKER")
    print('='*60)
    
    t0 = time.time()
    path = rrt_connect(start, goal, neural_checker, step_size=0.15, max_iter=3000)
    plan_time = time.time() - t0
    
    if path is None:
        print("\nPlanning failed!")
        input("Press Enter to exit...")
        p.disconnect()
        return
    
    print(f"\n✓ Planning time: {plan_time:.2f}s")
    
    # ── Validate with geometric checker ──
    print(f"\n{'='*60}")
    print("VALIDATING WITH GEOMETRIC CHECKER")
    print('='*60)
    
    n_invalid = 0
    for i, config in enumerate(path):
        if geometric_collision(config):
            n_invalid += 1
            print(f"  ⚠️  Waypoint {i}: COLLISION (geometric)")
    
    if n_invalid == 0:
        print("  ✓ All waypoints validated collision-free!")
    else:
        print(f"  ⚠️  {n_invalid}/{len(path)} waypoints in collision (geometric)")
    
    # ── Execute path ──
    input("\n>>> Press Enter to execute path...")
    
    print(f"\n{'='*60}")
    print("EXECUTING PATH")
    print('='*60)
    
    smooth_path = interpolate_path(path, max_step=0.03)
    print(f"Interpolated: {len(path)} → {len(smooth_path)} waypoints")
    
    set_robot_color(panda_id, COLOR_PATH)
    for i, config in enumerate(smooth_path):
        set_robot_config(panda_id, config)
        print(f"\r  Progress: {i+1}/{len(smooth_path)}", end="")
        time.sleep(0.025)
    print()
    
    # ── Done ──
    set_robot_color(panda_id, COLOR_FREE)
    print("\n✓ Path execution complete!")
    
    input("\n>>> Press Enter to exit...")
    p.disconnect()


if __name__ == "__main__":
    main()