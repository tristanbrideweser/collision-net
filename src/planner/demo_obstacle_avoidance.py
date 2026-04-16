"""
demo_obstacle_avoidance.py — Demo that clearly shows obstacle avoidance
"""

import sys
import types
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
import random
import time
import argparse
import pickle

# MPiNets shims
for _k in list(sys.modules.keys()):
    if "mpinets" in _k or "geometrout" in _k or "robofin" in _k:
        del sys.modules[_k]

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

for cls, mod in [(SE3, "geometrout.transform"), (SO3, "geometrout.transform"),
                 (Cuboid, "geometrout.primitive"), (Cylinder, "geometrout.primitive"),
                 (Sphere, "geometrout.primitive"), (PlanningProblem, "mpinets.mpinets_types"),
                 (FrankaRobot, "robofin.robots")]:
    cls.__module__ = mod
    sys.modules[mod].__dict__[cls.__name__] = cls
sys.modules["mpinets.utils"].PlanningProblem = PlanningProblem

import pybullet as p
import pybullet_data
from collision_detector import JOINT_LIMITS, init_collision_checker, in_collision as geometric_collision
from nn_collision_detector import NeuralCollisionChecker

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])

def rotation_matrix_to_quaternion(R):
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
    colors = {"Cuboid": [0.6, 0.3, 0.3, 0.9], "Cylinder": [0.3, 0.5, 0.6, 0.9], "Sphere": [0.4, 0.6, 0.3, 0.9]}
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

def sample_box_surface(half_extents, n=200):
    hx, hy, hz = half_extents
    pts = []
    for _ in range(n):
        face = random.randint(0, 5)
        if face == 0: pts.append([hx, random.uniform(-hy, hy), random.uniform(-hz, hz)])
        elif face == 1: pts.append([-hx, random.uniform(-hy, hy), random.uniform(-hz, hz)])
        elif face == 2: pts.append([random.uniform(-hx, hx), hy, random.uniform(-hz, hz)])
        elif face == 3: pts.append([random.uniform(-hx, hx), -hy, random.uniform(-hz, hz)])
        elif face == 4: pts.append([random.uniform(-hx, hx), random.uniform(-hy, hy), hz])
        else: pts.append([random.uniform(-hx, hx), random.uniform(-hy, hy), -hz])
    return np.array(pts)

def sample_cylinder_surface(radius, height, n=200):
    pts = []
    for _ in range(n):
        if random.random() < 0.7:
            theta = random.uniform(0, 2 * np.pi)
            z = random.uniform(-height / 2, height / 2)
            pts.append([radius * np.cos(theta), radius * np.sin(theta), z])
        else:
            r = radius * np.sqrt(random.random())
            theta = random.uniform(0, 2 * np.pi)
            z = height / 2 if random.random() < 0.5 else -height / 2
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(pts)

def sample_sphere_surface(radius, n=200):
    pts = []
    for _ in range(n):
        phi = random.uniform(0, 2 * np.pi)
        cos_theta = random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        pts.append([radius * sin_theta * np.cos(phi), radius * sin_theta * np.sin(phi), radius * cos_theta])
    return np.array(pts)

def extract_point_cloud(obstacle_ids, n_points=2048):
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
            body_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            pts = (body_mat @ pts.T).T + np.array(pos)
            all_points.append(pts)
    if not all_points:
        return np.zeros((n_points, 3), dtype=np.float32)
    all_points = np.concatenate(all_points, axis=0)
    idx = np.random.choice(len(all_points), n_points, replace=len(all_points) < n_points)
    return all_points[idx].astype(np.float32)

def set_robot_config(panda_id, config):
    for i in range(7):
        p.resetJointState(panda_id, i, config[i])

def set_robot_color(panda_id, color):
    for link_id in range(-1, 11):
        p.changeVisualShape(panda_id, link_id, rgbaColor=color)

class RRTNode:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None

def rrt_connect(start, goal, checker, step_size=0.15, max_iter=5000):
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
        q_rand = tree_b[0].conf if random.random() < 0.05 else sample_random()
        q_near_a = nearest(q_rand, tree_a)
        q_new = steer(q_near_a.conf, q_rand)
        
        if q_new is not None and edge_free(q_near_a.conf, q_new):
            node_new = RRTNode(q_new)
            node_new.parent = q_near_a
            tree_a.append(node_new)
            
            q_near_b = nearest(q_new, tree_b)
            q_connect = connect(q_near_b.conf, q_new)
            
            if q_connect is not None:
                node_connect = RRTNode(q_connect)
                node_connect.parent = q_near_b
                tree_b.append(node_connect)
                
                if np.linalg.norm(q_connect - q_new) < 1e-3:
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
                    print(f"  Path found: {i+1} iters, {len(path)} waypoints")
                    return path
        
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped
        
        if (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}...")
    
    print(f"  Failed after {max_iter} iterations")
    return None

def interpolate_path(path, max_step=0.02):
    result = [path[0]]
    for i in range(1, len(path)):
        dist = np.linalg.norm(path[i] - path[i-1])
        n = max(2, int(np.ceil(dist / max_step)))
        for j in range(1, n + 1):
            t = j / n
            result.append((1 - t) * path[i-1] + t * path[i])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../model/checkpoints/best.pt")
    parser.add_argument("--pkl", default="../data/envs/hybrid_solvable_problems.pkl")
    parser.add_argument("--scene-type", default="cubby")  # cubby has more obstacles
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.4)  # more conservative
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*60)
    print("OBSTACLE AVOIDANCE DEMO")
    print("="*60)
    
    print(f"\nLoading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        all_problems = pickle.load(f)
    
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
    
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(1.5, 60, -20, [0.4, 0, 0.5])
    
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    obstacle_ids = spawn_obstacles(prob.obstacles)
    print(f"Spawned {len(obstacle_ids)} obstacles")
    
    point_cloud = extract_point_cloud(obstacle_ids, n_points=2048)
    print(f"Point cloud: {point_cloud.shape}")
    
    neural_checker = NeuralCollisionChecker(
        scene_point_cloud=point_cloud,
        model_path=args.model,
        threshold=args.threshold,
    )
    
    init_collision_checker(panda_id=panda_id, obstacle_ids=obstacle_ids, table_id=None)
    
    # Use OPPOSITE SIDES of workspace to force obstacle avoidance
    # Left side start, right side goal
    START_CANDIDATES = [
        np.array([1.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785]),   # far left
        np.array([0.8, -0.3, 0.0, -1.8, 0.0, 1.5, 0.785]),   # left
        np.array([0.6, 0.0, 0.0, -1.5, 0.0, 1.7, 0.785]),    # left forward
    ]
    
    GOAL_CANDIDATES = [
        np.array([-1.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785]),  # far right
        np.array([-0.8, -0.3, 0.0, -1.8, 0.0, 1.5, 0.785]),  # right
        np.array([-0.6, 0.0, 0.0, -1.5, 0.0, 1.7, 0.785]),   # right forward
    ]
    
    # Find collision-free start
    start = None
    for candidate in START_CANDIDATES:
        if not neural_checker.in_collision(candidate):
            start = candidate
            print(f"Found valid start")
            break
    
    if start is None:
        # Fallback to home
        start = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        print("Using home as start")
    
    # Find collision-free goal on OPPOSITE side
    goal = None
    for candidate in GOAL_CANDIDATES:
        if not neural_checker.in_collision(candidate):
            goal = candidate
            print(f"Found valid goal on opposite side")
            break
    
    if goal is None:
        # Sample random goal on right side
        print("Sampling goal on right side of workspace...")
        for _ in range(500):
            candidate = np.array([
                random.uniform(-1.5, -0.3),  # negative = right side
                random.uniform(-1.0, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-2.5, -0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(1.0, 2.5),
                random.uniform(0.0, 1.5),
            ])
            # Clamp to joint limits
            candidate = np.clip(candidate, JOINT_LOWER, JOINT_UPPER)
            if not neural_checker.in_collision(candidate):
                goal = candidate
                break
    
    if goal is None:
        print("Could not find goal - using fallback")
        goal = np.array([-0.5, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    # Show straight-line would collide
    print("\n" + "-"*40)
    print("Checking if straight-line path would collide...")
    n_check = 50
    straight_path = np.linspace(start, goal, n_check)
    collisions = neural_checker.batch_in_collision(straight_path)
    n_col = collisions.sum()
    print(f"  Straight line: {n_col}/{n_check} configs in collision ({100*n_col/n_check:.0f}%)")
    if n_col > 0:
        print("  → RRT must find a path AROUND the obstacles!")
    print("-"*40)
    
    set_robot_config(panda_id, start)
    set_robot_color(panda_id, [0.2, 0.8, 0.2, 1.0])
    
    print(f"\nStart (left side):  {np.round(start[:3], 2)}...")
    print(f"Goal (right side):  {np.round(goal[:3], 2)}...")
    
    # Show goal position briefly
    print("\nShowing goal position (yellow)...")
    set_robot_color(panda_id, [0.9, 0.8, 0.2, 1.0])
    set_robot_config(panda_id, goal)
    time.sleep(1.5)
    
    # Back to start
    set_robot_color(panda_id, [0.2, 0.8, 0.2, 1.0])
    set_robot_config(panda_id, start)
    
    input("\n>>> Press Enter to plan (should avoid obstacles)...")
    
    print("\nPlanning with neural collision checker...")
    t0 = time.time()
    path = rrt_connect(start, goal, neural_checker, step_size=0.12, max_iter=5000)
    print(f"Time: {time.time() - t0:.2f}s")
    
    if path is None:
        print("\nPlanning failed! Try different seed or scene.")
        input("Press Enter to exit...")
        p.disconnect()
        return
    
    # Validate
    print("\nValidating with geometric checker...")
    n_bad = sum(1 for c in path if geometric_collision(c))
    if n_bad == 0:
        print("  ✓ All waypoints valid!")
    else:
        print(f"  ⚠️ {n_bad}/{len(path)} in collision")
    
    input("\n>>> Press Enter to execute (watch for obstacle avoidance)...")
    
    smooth = interpolate_path(path, max_step=0.02)
    print(f"Executing {len(smooth)} waypoints...")
    set_robot_color(panda_id, [0.2, 0.5, 0.9, 1.0])
    
    for i, c in enumerate(smooth):
        set_robot_config(panda_id, c)
        print(f"\r  {i+1}/{len(smooth)}", end="")
        time.sleep(0.03)
    print()
    
    set_robot_color(panda_id, [0.2, 0.8, 0.2, 1.0])
    print("\n✓ Done! The robot should have moved AROUND the obstacles.")
    
    input("\nPress Enter to exit...")
    p.disconnect()

if __name__ == "__main__":
    main()
