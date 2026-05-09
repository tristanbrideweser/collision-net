"""
run_benchmark.py — Head-to-head comparison: Geometric (PyBullet) vs Neural (CollisionNet).

Fills in the LaTeX table:
  - Throughput (queries/s)
  - Avg. query latency (us)
  - Success rate (%) -- planning success
  - Validation accuracy (%) -- per-config agreement vs geometric ground truth
  - False negative rate (%) -- neural says "free" when actually in collision (the dangerous case)
  - C-space collision rate -- prior on uniform random samples in this scene set
  - Mean planning time (s)
  - Path length (rad) -- sum of L2 norms between consecutive joint configs
  - Mean clearance (cm) -- min distance from robot to nearest obstacle along path

Usage (from repo root):
    cd scripts
    python run_benchmark.py \
        --pkl ../data/envs/hybrid_solvable_problems.pkl \
        --model ../checkpoints/best.pt \
        --n-scenes 50 \
        --problems-per-scene 5 \
        --threshold 0.5 \
        --seed 0 \
        --out benchmark_results.json

The script is GUI-less (p.DIRECT) for speed.
"""

import sys
import os
import types
import time
import json
import pickle
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
import pybullet as p
import pybullet_data

# ─────────────────────────────────────────────────────────────
# 1. MPiNets shims (so the pkl unpickles without the full stack)
# ─────────────────────────────────────────────────────────────

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
                 (Sphere, "geometrout.primitive"),
                 (PlanningProblem, "mpinets.mpinets_types"),
                 (FrankaRobot, "robofin.robots")]:
    cls.__module__ = mod
    sys.modules[mod].__dict__[cls.__name__] = cls
sys.modules["mpinets.utils"].PlanningProblem = PlanningProblem

# ─────────────────────────────────────────────────────────────
# 2. Project imports
# ─────────────────────────────────────────────────────────────

# Make sure src/ is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from planner.collision_detector import (
    JOINT_LIMITS,
    init_collision_checker,
    update_obstacles,
    in_collision as geometric_in_collision,
    reset_collision_count,
    get_collision_count,
)
from planner.nn_collision_detector import NeuralCollisionChecker

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])

# ─────────────────────────────────────────────────────────────
# 3. Scene reconstruction (lifted from demo_neural_rrt.py, trimmed)
# ─────────────────────────────────────────────────────────────

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
    pose = getattr(obs, "_pose", None)
    if pose is None:
        return [0, 0, 0], [0, 0, 0, 1]
    xyz = getattr(pose, "_xyz", None)
    pos = list(xyz) if xyz is not None else [0, 0, 0]
    so3 = getattr(pose, "_so3", None)
    if so3 is not None:
        mat = getattr(so3, "matrix", None)
        if mat is not None and hasattr(mat, "shape") and mat.shape == (3, 3):
            orn = rotation_matrix_to_quaternion(mat)
        else:
            orn = [0, 0, 0, 1]
    else:
        orn = [0, 0, 0, 1]
    return pos, orn


def spawn_obstacles(obstacles):
    obj_ids = []
    for obs in obstacles:
        pos, orn = get_obstacle_pose(obs)
        t = type(obs).__name__
        if t == "Cuboid":
            dims = getattr(obs, "_dims", np.ones(3))
            half = [float(d) / 2 for d in dims]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        elif t == "Cylinder":
            r = getattr(obs, "radius", 0.05)
            h = getattr(obs, "height", 0.1)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
        elif t == "Sphere":
            r = getattr(obs, "radius", 0.05)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        else:
            continue
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            basePosition=pos,
            baseOrientation=orn,
        )
        obj_ids.append(body_id)
    return obj_ids


def sample_box_surface(half_extents, n=300):
    hx, hy, hz = half_extents
    pts = []
    for _ in range(n):
        face = random.randint(0, 5)
        if face == 0:   pts.append([hx,  random.uniform(-hy, hy), random.uniform(-hz, hz)])
        elif face == 1: pts.append([-hx, random.uniform(-hy, hy), random.uniform(-hz, hz)])
        elif face == 2: pts.append([random.uniform(-hx, hx),  hy, random.uniform(-hz, hz)])
        elif face == 3: pts.append([random.uniform(-hx, hx), -hy, random.uniform(-hz, hz)])
        elif face == 4: pts.append([random.uniform(-hx, hx), random.uniform(-hy, hy),  hz])
        else:           pts.append([random.uniform(-hx, hx), random.uniform(-hy, hy), -hz])
    return np.array(pts)


def sample_cylinder_surface(radius, height, n=300):
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


def sample_sphere_surface(radius, n=300):
    pts = []
    for _ in range(n):
        phi = random.uniform(0, 2 * np.pi)
        cos_theta = random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        pts.append([radius * sin_theta * np.cos(phi),
                    radius * sin_theta * np.sin(phi),
                    radius * cos_theta])
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


# ─────────────────────────────────────────────────────────────
# 4. RRT-Connect (matches demo_neural_rrt.py, with timing hooks)
# ─────────────────────────────────────────────────────────────

class RRTNode:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None


def rrt_connect_with_stats(start, goal, single_check_fn, batch_check_fn,
                           step_size=0.15, max_iter=3000):
    """
    RRT-Connect that records (n_queries, planning_time).
    `single_check_fn(q) -> bool` and `batch_check_fn(np.ndarray (N,7)) -> np.ndarray (N,) bool`.
    Returns (path | None, stats_dict).
    """
    stats = {"single_queries": 0, "batch_queries": 0, "total_configs_checked": 0}

    def single_check(q):
        stats["single_queries"] += 1
        stats["total_configs_checked"] += 1
        return single_check_fn(q)

    def batch_check(qs):
        stats["batch_queries"] += 1
        stats["total_configs_checked"] += len(qs)
        return batch_check_fn(qs)

    def sample_random():
        return np.array([random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])

    def nearest(q, tree):
        dists = [np.linalg.norm(q - n.conf) for n in tree]
        return tree[int(np.argmin(dists))]

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
        return not batch_check(waypoints).any()

    def connect(q_from, q_to):
        dist = np.linalg.norm(q_to - q_from)
        n_steps = max(2, int(np.ceil(dist / (step_size / 2))))
        waypoints = np.linspace(q_from, q_to, n_steps)
        collisions = batch_check(waypoints)
        if not collisions.any():
            return q_to
        first_col = int(collisions.argmax())
        if first_col == 0:
            return None
        return waypoints[first_col - 1]

    t0 = time.perf_counter()

    if single_check(start) or single_check(goal):
        return None, {**stats, "planning_time": time.perf_counter() - t0,
                      "iterations": 0, "reason": "endpoint_in_collision"}

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
                    stats["planning_time"] = time.perf_counter() - t0
                    stats["iterations"] = i + 1
                    stats["reason"] = "success"
                    return path, stats

        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    stats["planning_time"] = time.perf_counter() - t0
    stats["iterations"] = max_iter
    stats["reason"] = "max_iter"
    return None, stats


# ─────────────────────────────────────────────────────────────
# 5. Path-quality metrics
# ─────────────────────────────────────────────────────────────

def path_length_rad(path: List[np.ndarray]) -> float:
    """Sum of L2 norms between consecutive 7-D joint configs."""
    if path is None or len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def interpolate_path(path, max_step=0.03):
    if path is None:
        return []
    result = [path[0]]
    for i in range(1, len(path)):
        dist = np.linalg.norm(path[i] - path[i - 1])
        n = max(2, int(np.ceil(dist / max_step)))
        for j in range(1, n + 1):
            t = j / n
            result.append((1 - t) * path[i - 1] + t * path[i])
    return result


def path_clearance_cm(path, panda_id, obstacle_ids, max_dist=2.0) -> Tuple[float, float]:
    """
    Returns (mean_clearance_cm, min_clearance_cm) along densely-interpolated path.
    Uses PyBullet getClosestPoints between robot and each obstacle.
    """
    if path is None or len(path) < 2:
        return float("nan"), float("nan")

    dense = interpolate_path(path, max_step=0.03)
    per_step_min = []

    for q in dense:
        for j in range(7):
            p.resetJointState(panda_id, j, q[j])
        p.performCollisionDetection()

        step_min = max_dist
        for obs_id in obstacle_ids:
            cps = p.getClosestPoints(panda_id, obs_id, distance=max_dist)
            for cp in cps:
                # cp[3] is link index on robot; ignore base (link 0)
                if cp[3] == 0:
                    continue
                d = cp[8]  # contactDistance (positive = separation, negative = penetration)
                if d < step_min:
                    step_min = d
        per_step_min.append(step_min)

    arr = np.array(per_step_min) * 100.0  # m -> cm
    return float(arr.mean()), float(arr.min())


def path_geometric_validity(path, dense=True) -> Tuple[int, int]:
    """Returns (n_in_collision, n_total) using the geometric checker."""
    if path is None or len(path) < 1:
        return 0, 0
    pts = interpolate_path(path) if dense else path
    n_bad = sum(1 for q in pts if geometric_in_collision(q))
    return n_bad, len(pts)


# ─────────────────────────────────────────────────────────────
# 6. Throughput / latency micro-benchmarks (per-scene)
# ─────────────────────────────────────────────────────────────

def benchmark_query_speed(check_fn, n_queries=500, warmup=20):
    """
    Time `n_queries` single calls to `check_fn(q)` with random in-limit configs.
    Returns (queries_per_second, mean_latency_us).
    """
    rng = np.random.default_rng(0)
    configs = rng.uniform(JOINT_LOWER, JOINT_UPPER, size=(n_queries + warmup, 7))

    # Warmup
    for q in configs[:warmup]:
        _ = check_fn(q)

    t0 = time.perf_counter()
    for q in configs[warmup:]:
        _ = check_fn(q)
    elapsed = time.perf_counter() - t0

    qps = n_queries / elapsed
    latency_us = (elapsed / n_queries) * 1e6
    return qps, latency_us


# ─────────────────────────────────────────────────────────────
# 7. Validation accuracy & FNR (neural vs geometric ground truth)
# ─────────────────────────────────────────────────────────────

def validation_metrics(neural_checker, n_samples=500):
    """
    Sample uniform-random configs in this scene, label with both checkers,
    and report agreement metrics with geometric as ground truth.
    """
    rng = np.random.default_rng(1)
    configs = rng.uniform(JOINT_LOWER, JOINT_UPPER, size=(n_samples, 7))

    geom_labels = np.array([geometric_in_collision(q) for q in configs])
    neural_labels = neural_checker.batch_in_collision(configs)

    tp = int(np.sum(neural_labels & geom_labels))
    tn = int(np.sum(~neural_labels & ~geom_labels))
    fp = int(np.sum(neural_labels & ~geom_labels))
    fn = int(np.sum(~neural_labels & geom_labels))

    n_pos = int(geom_labels.sum())
    n_neg = int((~geom_labels).sum())

    return {
        "n_samples": n_samples,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n_samples if n_samples else float("nan"),
        # FNR = FN / (FN + TP)  -- neural says "free" when actually colliding.
        # This is the safety-critical error.
        "false_negative_rate": fn / n_pos if n_pos > 0 else float("nan"),
        "false_positive_rate": fp / n_neg if n_neg > 0 else float("nan"),
        "cspace_collision_rate": n_pos / n_samples,  # prior on this scene
    }


# ─────────────────────────────────────────────────────────────
# 8. Per-scene driver
# ─────────────────────────────────────────────────────────────

def collect_scenes(pkl_path, n_scenes):
    """Walk all (scene_type, category) buckets, return up to n_scenes problems."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    flat = []
    for scene_type, cats in data.items():
        if not isinstance(cats, dict):
            continue
        for cat_name, probs in cats.items():
            if not isinstance(probs, list):
                continue
            for idx, prob in enumerate(probs):
                if prob.obstacles is None or len(prob.obstacles) == 0:
                    continue
                flat.append((scene_type, cat_name, idx, prob))

    random.Random(0).shuffle(flat)
    return flat[:n_scenes]


def sample_start_goal(neural_checker, max_attempts=500):
    """Find a (start, goal) pair where BOTH checkers agree both are collision-free."""
    rng = np.random.default_rng()
    start = goal = None
    for _ in range(max_attempts):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        if not neural_checker.in_collision(q) and not geometric_in_collision(q):
            if start is None:
                start = q
            elif np.linalg.norm(q - start) > 1.0:
                goal = q
                break
    return start, goal


def run_scene(prob, model_path, threshold, problems_per_scene,
              n_validation_samples, n_throughput_queries):
    """Reset PyBullet, build scene, run all measurements. Returns dict."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    obstacle_ids = spawn_obstacles(prob.obstacles)
    init_collision_checker(panda_id=panda_id, obstacle_ids=obstacle_ids, table_id=None)

    point_cloud = extract_point_cloud(obstacle_ids, n_points=2048)

    neural_checker = NeuralCollisionChecker(
        scene_point_cloud=point_cloud,
        panda_id=panda_id,
        model_path=model_path,
        threshold=threshold,
    )

    # --- Validation accuracy / FNR ---
    val = validation_metrics(neural_checker, n_samples=n_validation_samples)

    # --- Throughput / latency (per-checker) ---
    geom_qps, geom_lat = benchmark_query_speed(geometric_in_collision, n_queries=n_throughput_queries)
    neural_qps, neural_lat = benchmark_query_speed(neural_checker.in_collision, n_queries=n_throughput_queries)

    # --- Planning, both checkers, multiple problems ---
    planning_records = []
    for prob_idx in range(problems_per_scene):
        start, goal = sample_start_goal(neural_checker)
        if start is None or goal is None:
            continue

        # Geometric planner
        path_geom, stats_geom = rrt_connect_with_stats(
            start, goal,
            single_check_fn=geometric_in_collision,
            batch_check_fn=lambda qs: np.array([geometric_in_collision(q) for q in qs]),
        )
        geom_clearance_mean, geom_clearance_min = path_clearance_cm(
            path_geom, panda_id, obstacle_ids
        ) if path_geom is not None else (float("nan"), float("nan"))

        # Neural planner
        path_neur, stats_neur = rrt_connect_with_stats(
            start, goal,
            single_check_fn=neural_checker.in_collision,
            batch_check_fn=neural_checker.batch_in_collision,
        )
        neur_bad, neur_total = path_geometric_validity(path_neur)
        neur_clearance_mean, neur_clearance_min = path_clearance_cm(
            path_neur, panda_id, obstacle_ids
        ) if path_neur is not None else (float("nan"), float("nan"))

        planning_records.append({
            "geom": {
                "success": path_geom is not None,
                "time_s": stats_geom["planning_time"],
                "iterations": stats_geom["iterations"],
                "configs_checked": stats_geom["total_configs_checked"],
                "path_len_rad": path_length_rad(path_geom),
                "clearance_mean_cm": geom_clearance_mean,
                "clearance_min_cm": geom_clearance_min,
            },
            "neural": {
                "success": path_neur is not None,
                "time_s": stats_neur["planning_time"],
                "iterations": stats_neur["iterations"],
                "configs_checked": stats_neur["total_configs_checked"],
                "path_len_rad": path_length_rad(path_neur),
                "clearance_mean_cm": neur_clearance_mean,
                "clearance_min_cm": neur_clearance_min,
                # Geometric validity: did the neural planner produce a path that
                # is actually collision-free under the ground-truth checker?
                "geometrically_valid": (neur_bad == 0) if path_neur is not None else False,
                "n_bad_waypoints": neur_bad,
                "n_total_waypoints": neur_total,
            },
        })

    return {
        "validation": val,
        "throughput": {
            "geom_qps": geom_qps, "geom_latency_us": geom_lat,
            "neural_qps": neural_qps, "neural_latency_us": neural_lat,
        },
        "planning": planning_records,
    }


# ─────────────────────────────────────────────────────────────
# 9. Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate(scene_results):
    """Combine per-scene records into the final table values."""
    # --- Throughput / latency: average across scenes ---
    geom_qps = np.mean([s["throughput"]["geom_qps"] for s in scene_results])
    geom_lat = np.mean([s["throughput"]["geom_latency_us"] for s in scene_results])
    neur_qps = np.mean([s["throughput"]["neural_qps"] for s in scene_results])
    neur_lat = np.mean([s["throughput"]["neural_latency_us"] for s in scene_results])

    # --- Validation: pool across scenes ---
    tp = sum(s["validation"]["tp"] for s in scene_results)
    tn = sum(s["validation"]["tn"] for s in scene_results)
    fp = sum(s["validation"]["fp"] for s in scene_results)
    fn = sum(s["validation"]["fn"] for s in scene_results)
    n_total = tp + tn + fp + fn
    n_pos = tp + fn
    n_neg = tn + fp

    val_acc = (tp + tn) / n_total if n_total else float("nan")
    fnr = fn / n_pos if n_pos else float("nan")
    cspace_collision_rate = n_pos / n_total if n_total else float("nan")

    # --- Planning ---
    geom_records = [r["geom"] for s in scene_results for r in s["planning"]]
    neur_records = [r["neural"] for s in scene_results for r in s["planning"]]

    def stats(records, key, only_success=True):
        if only_success:
            vals = [r[key] for r in records if r["success"] and not np.isnan(r[key])]
        else:
            vals = [r[key] for r in records if not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    geom_success_rate = np.mean([r["success"] for r in geom_records]) if geom_records else float("nan")
    neur_success_rate = np.mean([r["success"] for r in neur_records]) if neur_records else float("nan")

    geom_time_mean, geom_time_std = stats(geom_records, "time_s")
    neur_time_mean, neur_time_std = stats(neur_records, "time_s")
    geom_path_mean, geom_path_std = stats(geom_records, "path_len_rad")
    neur_path_mean, neur_path_std = stats(neur_records, "path_len_rad")
    geom_clr_mean, geom_clr_std = stats(geom_records, "clearance_mean_cm")
    neur_clr_mean, neur_clr_std = stats(neur_records, "clearance_mean_cm")

    # Fraction of neural-planner paths that are actually collision-free under the
    # ground-truth checker (path-level safety).
    neur_paths_valid = [r["geometrically_valid"] for r in neur_records if r["success"]]
    neur_path_valid_rate = float(np.mean(neur_paths_valid)) if neur_paths_valid else float("nan")

    return {
        "throughput": {
            "geom_qps": float(geom_qps),
            "neural_qps": float(neur_qps),
            "geom_latency_us": float(geom_lat),
            "neural_latency_us": float(neur_lat),
        },
        "safety": {
            "validation_accuracy": float(val_acc),
            "false_negative_rate": float(fnr),
            "cspace_collision_rate_geom": float(cspace_collision_rate),
            "n_validation_samples": int(n_total),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "neural_path_geometric_validity_rate": neur_path_valid_rate,
        },
        "planning": {
            "geom_success_rate": float(geom_success_rate),
            "neural_success_rate": float(neur_success_rate),
            "geom_time_mean_s": geom_time_mean,
            "geom_time_std_s": geom_time_std,
            "neural_time_mean_s": neur_time_mean,
            "neural_time_std_s": neur_time_std,
            "geom_path_len_mean_rad": geom_path_mean,
            "geom_path_len_std_rad": geom_path_std,
            "neural_path_len_mean_rad": neur_path_mean,
            "neural_path_len_std_rad": neur_path_std,
            "geom_clearance_mean_cm": geom_clr_mean,
            "geom_clearance_std_cm": geom_clr_std,
            "neural_clearance_mean_cm": neur_clr_mean,
            "neural_clearance_std_cm": neur_clr_std,
            "n_problems": len(geom_records),
        },
    }


# ─────────────────────────────────────────────────────────────
# 10. Pretty-print the final LaTeX table
# ─────────────────────────────────────────────────────────────

def format_table(agg):
    t = agg["throughput"]
    s = agg["safety"]
    p_ = agg["planning"]

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Metric':<35} {'Geometric':>15} {'Neural':>15}")
    print("-" * 70)
    print(f"{'Throughput (queries/s)':<35} {t['geom_qps']:>15.1f} {t['neural_qps']:>15.1f}")
    print(f"{'Avg. query latency (us)':<35} {t['geom_latency_us']:>15.1f} {t['neural_latency_us']:>15.1f}")
    print(f"{'Planning success rate (%)':<35} {p_['geom_success_rate']*100:>15.1f} {p_['neural_success_rate']*100:>15.1f}")
    print("-" * 70)
    print("Safety & Accuracy")
    print(f"{'Validation accuracy (%)':<35} {'100.0':>15} {s['validation_accuracy']*100:>15.2f}")
    print(f"{'False negative rate (%)':<35} {'0.0':>15} {s['false_negative_rate']*100:>15.2f}")
    print(f"{'C-space collision rate (%)':<35} {s['cspace_collision_rate_geom']*100:>15.2f} {s['cspace_collision_rate_geom']*100:>15.2f}")
    print(f"{'Neural-path validity rate (%)':<35} {'-':>15} {s['neural_path_geometric_validity_rate']*100:>15.2f}")
    print("-" * 70)
    print("Planning Performance")
    print(f"{'Mean planning time (s)':<35} "
          f"{p_['geom_time_mean_s']:>10.2f}\u00b1{p_['geom_time_std_s']:<3.2f} "
          f"{p_['neural_time_mean_s']:>10.2f}\u00b1{p_['neural_time_std_s']:<3.2f}")
    print(f"{'Path length (rad)':<35} "
          f"{p_['geom_path_len_mean_rad']:>10.2f}\u00b1{p_['geom_path_len_std_rad']:<3.2f} "
          f"{p_['neural_path_len_mean_rad']:>10.2f}\u00b1{p_['neural_path_len_std_rad']:<3.2f}")
    print(f"{'Mean clearance (cm)':<35} "
          f"{p_['geom_clearance_mean_cm']:>10.2f}\u00b1{p_['geom_clearance_std_cm']:<3.2f} "
          f"{p_['neural_clearance_mean_cm']:>10.2f}\u00b1{p_['neural_clearance_std_cm']:<3.2f}")
    print("=" * 70)
    print(f"(Aggregated over {p_['n_problems']} planning problems, "
          f"{s['n_validation_samples']} validation samples)")


# ─────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True,
                    help="Path to MPiNets problems pkl (e.g. data/envs/hybrid_solvable_problems.pkl)")
    ap.add_argument("--model", required=True,
                    help="Path to CollisionNet checkpoint (e.g. checkpoints/best.pt)")
    ap.add_argument("--n-scenes", type=int, default=50)
    ap.add_argument("--problems-per-scene", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-validation-samples", type=int, default=500,
                    help="Random configs per scene used to compute accuracy/FNR")
    ap.add_argument("--n-throughput-queries", type=int, default=300,
                    help="Per-scene timing budget for throughput micro-bench")
    ap.add_argument("--out", type=str, default="benchmark_results.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\nLoading scenes from {args.pkl}...")
    scenes = collect_scenes(args.pkl, args.n_scenes)
    print(f"Selected {len(scenes)} scenes")

    p.connect(p.DIRECT)

    scene_results = []
    t_total = time.perf_counter()

    for scene_idx, (stype, cat, idx, prob) in enumerate(scenes):
        print(f"\n[{scene_idx+1}/{len(scenes)}] {stype}/{cat}#{idx} "
              f"({len(prob.obstacles)} obstacles)")
        try:
            res = run_scene(
                prob,
                model_path=args.model,
                threshold=args.threshold,
                problems_per_scene=args.problems_per_scene,
                n_validation_samples=args.n_validation_samples,
                n_throughput_queries=args.n_throughput_queries,
            )
            scene_results.append(res)
            v = res["validation"]
            tp = res["throughput"]
            print(f"  val_acc={v['accuracy']*100:.1f}%  fnr={v['false_negative_rate']*100:.1f}%  "
                  f"geom={tp['geom_qps']:.0f}q/s  neural={tp['neural_qps']:.0f}q/s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    p.disconnect()

    if not scene_results:
        print("No scenes succeeded; bailing.")
        return

    agg = aggregate(scene_results)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal benchmark wall-time: {elapsed/60:.1f} min")

    format_table(agg)

    with open(args.out, "w") as f:
        json.dump({"aggregate": agg, "args": vars(args)}, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()