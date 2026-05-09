"""
make_comparison_video.py — Side-by-side video: Geometric vs Neural RRT-Connect.

For each of N scenes:
  1. Build scene in PyBullet DIRECT
  2. Sample one start/goal pair (collision-free under BOTH checkers)
  3. Plan with each checker (same seed, same problem)
  4. Render execution frames via getCameraImage
  5. ffmpeg into side-by-side MP4 with labels
  6. Concat all scene clips into one montage

Usage (run from repo root after activating venv):
    python scripts/make_comparison_video.py \
        --pkl src/data/envs/hybrid_solvable_problems.pkl \
        --model src/model/checkpoints/generalist.pth \
        --out comparison_montage.mp4

Requires ffmpeg on PATH.
"""

import sys
import os
import types
import time
import pickle
import random
import argparse
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Union, List

import numpy as np
import pybullet as p
import pybullet_data

# ─────────────────────────────────────────────────────────────
# 1. MPiNets shims
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from planner.collision_detector import (
    JOINT_LIMITS,
    init_collision_checker,
    in_collision as geometric_in_collision,
)
from planner.nn_collision_detector import NeuralCollisionChecker

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])


# ─────────────────────────────────────────────────────────────
# 3. Scene reconstruction (lifted from demo)
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
    """Spawn obstacles with light gray visuals (so the robot stands out)."""
    obj_ids = []
    color = [0.6, 0.6, 0.65, 0.9]
    for obs in obstacles:
        pos, orn = get_obstacle_pose(obs)
        t = type(obs).__name__
        if t == "Cuboid":
            dims = getattr(obs, "_dims", np.ones(3))
            half = [float(d) / 2 for d in dims]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        elif t == "Cylinder":
            r = getattr(obs, "radius", 0.05)
            h = getattr(obs, "height", 0.1)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
        elif t == "Sphere":
            r = getattr(obs, "radius", 0.05)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
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
# 4. RRT-Connect (matches benchmark)
# ─────────────────────────────────────────────────────────────

class RRTNode:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None


def rrt_connect(start, goal, single_check_fn, batch_check_fn,
                step_size=0.15, max_iter=3000, seed=0):
    rng_state = random.getstate()
    random.seed(seed)

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
        return not batch_check_fn(waypoints).any()

    def connect(q_from, q_to):
        dist = np.linalg.norm(q_to - q_from)
        n_steps = max(2, int(np.ceil(dist / (step_size / 2))))
        waypoints = np.linspace(q_from, q_to, n_steps)
        collisions = batch_check_fn(waypoints)
        if not collisions.any():
            return q_to
        first_col = int(collisions.argmax())
        if first_col == 0:
            return None
        return waypoints[first_col - 1]

    if single_check_fn(start) or single_check_fn(goal):
        random.setstate(rng_state)
        return None

    tree_a = [RRTNode(start)]
    tree_b = [RRTNode(goal)]
    swapped = False

    path = None
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
                    break

        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    random.setstate(rng_state)
    return path


def interpolate_path(path, max_step=0.02):
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


# ─────────────────────────────────────────────────────────────
# 5. Rendering
# ─────────────────────────────────────────────────────────────

def setup_camera(width, height, target=(0.3, 0.0, 0.4),
                 distance=1.8, yaw=45, pitch=-25):
    """Returns (view_matrix, proj_matrix) for getCameraImage."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.05, farVal=10.0,
    )
    return view, proj


def grab_frame(width, height, view, proj):
    """Returns RGB uint8 array (H, W, 3)."""
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,  # CPU renderer; works in DIRECT mode
        flags=p.ER_NO_SEGMENTATION_MASK,
    )
    rgba = np.reshape(rgba, (height, width, 4))
    return rgba[:, :, :3].astype(np.uint8)


def set_robot_config(panda_id, q):
    for i in range(7):
        p.resetJointState(panda_id, i, q[i])


def set_robot_color(panda_id, rgba):
    for link_id in range(-1, 11):
        p.changeVisualShape(panda_id, link_id, rgbaColor=rgba)


def render_execution(panda_id, path, frame_dir, width, height, view, proj,
                     hold_endpoints_frames=15):
    """Write PNG frames as the robot follows the path."""
    os.makedirs(frame_dir, exist_ok=True)
    dense = interpolate_path(path, max_step=0.02)
    frame_idx = 0

    # Hold start
    set_robot_config(panda_id, dense[0])
    for _ in range(hold_endpoints_frames):
        img = grab_frame(width, height, view, proj)
        save_png(img, os.path.join(frame_dir, f"frame_{frame_idx:05d}.png"))
        frame_idx += 1

    # Animate
    for q in dense:
        set_robot_config(panda_id, q)
        img = grab_frame(width, height, view, proj)
        save_png(img, os.path.join(frame_dir, f"frame_{frame_idx:05d}.png"))
        frame_idx += 1

    # Hold goal
    for _ in range(hold_endpoints_frames):
        img = grab_frame(width, height, view, proj)
        save_png(img, os.path.join(frame_dir, f"frame_{frame_idx:05d}.png"))
        frame_idx += 1

    return frame_idx


def save_png(img, path):
    """Minimal PNG writer using imageio if available, else PIL."""
    try:
        import imageio
        imageio.imwrite(path, img)
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


# ─────────────────────────────────────────────────────────────
# 6. ffmpeg helpers
# ─────────────────────────────────────────────────────────────

def frames_to_mp4(frame_dir, out_path, fps=30, label=None):
    """Encode PNG frames into an MP4 with optional top-left label."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
    ]
    if label:
        # Drawtext filter; uses default font on most Linux systems
        # Escape colons in label
        safe = label.replace(":", r"\:").replace("'", r"\\'")
        cmd += ["-vf",
                f"drawtext=text='{safe}':fontcolor=white:fontsize=28:"
                f"box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out_path]
    subprocess.run(cmd, check=True)


def stack_side_by_side(left_mp4, right_mp4, out_path):
    """hstack two MP4s. Assumes same height; ffmpeg pads if needed."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", left_mp4, "-i", right_mp4,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def concat_mp4s(input_paths, out_path):
    """Concatenate MP4s into a single video."""
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for pth in input_paths:
            f.write(f"file '{os.path.abspath(pth)}'\n")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy", out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_file)


# ─────────────────────────────────────────────────────────────
# 7. Per-scene driver
# ─────────────────────────────────────────────────────────────

def find_one_scene(all_problems, scene_type):
    """Find first solvable PlanningProblem for the given scene_type."""
    if scene_type not in all_problems:
        return None
    cats = all_problems[scene_type]
    if not isinstance(cats, dict):
        return None
    for cat_name in ["task_oriented", "neutral_start", "neutral_goal"]:
        probs = cats.get(cat_name, [])
        if isinstance(probs, list):
            for prob in probs:
                if prob.obstacles and len(prob.obstacles) > 0:
                    return prob
    return None


def sample_start_goal(neural_checker, max_attempts=500, min_separation=1.0):
    rng = np.random.default_rng()
    start = goal = None
    for _ in range(max_attempts):
        q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        if not neural_checker.in_collision(q) and not geometric_in_collision(q):
            if start is None:
                start = q
            elif np.linalg.norm(q - start) > min_separation:
                goal = q
                break
    return start, goal


def render_scene_pair(prob, scene_label, model_path, threshold,
                      width, height, fps, work_dir, seed=0):
    """Returns path to side-by-side MP4 for this scene, or None on failure."""
    print(f"\n=== Rendering scene: {scene_label} ===")

    # Fresh PyBullet world for this scene
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # silence GUI updates
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

    # Common start/goal
    random.seed(seed); np.random.seed(seed)
    start, goal = sample_start_goal(neural_checker)
    if start is None or goal is None:
        print(f"  Could not sample start/goal for {scene_label}; skipping.")
        return None
    print(f"  Start/goal sampled.")

    # Plan with each checker (same seed -> same RNG draws if checkers agree)
    print(f"  Planning (geometric)...")
    t0 = time.time()
    path_geom = rrt_connect(
        start, goal,
        single_check_fn=geometric_in_collision,
        batch_check_fn=lambda qs: np.array([geometric_in_collision(q) for q in qs]),
        seed=seed,
    )
    t_geom = time.time() - t0
    print(f"    {'OK' if path_geom else 'FAIL'}  ({t_geom:.2f}s)")

    print(f"  Planning (neural)...")
    t0 = time.time()
    path_neur = rrt_connect(
        start, goal,
        single_check_fn=neural_checker.in_collision,
        batch_check_fn=neural_checker.batch_in_collision,
        seed=seed,
    )
    t_neur = time.time() - t0
    print(f"    {'OK' if path_neur else 'FAIL'}  ({t_neur:.2f}s)")

    if path_geom is None or path_neur is None:
        print(f"  One planner failed; skipping {scene_label}.")
        return None

    # Camera setup (same view for both)
    view, proj = setup_camera(width, height)

    # Render geometric execution (blue robot)
    print(f"  Rendering geometric execution...")
    set_robot_color(panda_id, [0.2, 0.5, 0.9, 1.0])
    geom_frames = os.path.join(work_dir, f"{scene_label}_geom_frames")
    render_execution(panda_id, path_geom, geom_frames, width, height, view, proj)

    # Render neural execution (orange robot, same scene)
    print(f"  Rendering neural execution...")
    set_robot_color(panda_id, [0.95, 0.55, 0.15, 1.0])
    neur_frames = os.path.join(work_dir, f"{scene_label}_neural_frames")
    render_execution(panda_id, path_neur, neur_frames, width, height, view, proj)

    # Encode each side
    geom_mp4 = os.path.join(work_dir, f"{scene_label}_geom.mp4")
    neur_mp4 = os.path.join(work_dir, f"{scene_label}_neural.mp4")
    label_geom = f"Geometric (PyBullet)  |  {len(path_geom)} wpts  {t_geom:.2f}s"
    label_neur = f"Neural (Ours)  |  {len(path_neur)} wpts  {t_neur:.2f}s"

    print(f"  Encoding MP4s...")
    frames_to_mp4(geom_frames, geom_mp4, fps=fps, label=label_geom)
    frames_to_mp4(neur_frames, neur_mp4, fps=fps, label=label_neur)

    # Pad shorter side so hstack lines up (ffmpeg auto-pads with hstack but
    # frame counts can mismatch; simplest: re-encode so both have same duration
    # by tpad on the shorter video).
    sbs = os.path.join(work_dir, f"{scene_label}_sbs.mp4")
    pad_then_stack(geom_mp4, neur_mp4, sbs, fps=fps)

    # Cleanup frames (not the MP4s yet, we may want them)
    shutil.rmtree(geom_frames, ignore_errors=True)
    shutil.rmtree(neur_frames, ignore_errors=True)

    return sbs


def pad_then_stack(left_mp4, right_mp4, out_path, fps):
    """Stack two MP4s side-by-side, padding the shorter one with its last frame."""
    # Get durations
    def duration(mp4):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", mp4],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())

    d_left, d_right = duration(left_mp4), duration(right_mp4)
    d_max = max(d_left, d_right)
    pad_left = max(0.0, d_max - d_left)
    pad_right = max(0.0, d_max - d_right)

    # tpad clones the last frame for `stop_duration` seconds
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", left_mp4, "-i", right_mp4,
        "-filter_complex",
        f"[0:v]tpad=stop_mode=clone:stop_duration={pad_left}[L];"
        f"[1:v]tpad=stop_mode=clone:stop_duration={pad_right}[R];"
        f"[L][R]hstack=inputs=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-r", str(fps),
        out_path,
    ]
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scene-types", nargs="+",
                    default=["cubby", "tabletop", "dresser"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", type=str, default="video_work")
    ap.add_argument("--out", type=str, default="comparison_montage.mp4")
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    # Verify ffmpeg
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH.")
        sys.exit(1)

    os.makedirs(args.work_dir, exist_ok=True)

    print(f"Loading scenes from {args.pkl}...")
    with open(args.pkl, "rb") as f:
        all_problems = pickle.load(f)

    # PyBullet DIRECT (no display, getCameraImage uses TINY renderer)
    p.connect(p.DIRECT)

    sbs_clips = []
    for stype in args.scene_types:
        prob = find_one_scene(all_problems, stype)
        if prob is None:
            print(f"  No scene found for type '{stype}'; skipping.")
            continue
        sbs = render_scene_pair(
            prob, scene_label=stype,
            model_path=args.model,
            threshold=args.threshold,
            width=args.width, height=args.height, fps=args.fps,
            work_dir=args.work_dir,
            seed=args.seed,
        )
        if sbs is not None:
            sbs_clips.append(sbs)

    p.disconnect()

    if not sbs_clips:
        print("No scenes rendered successfully.")
        sys.exit(1)

    print(f"\nConcatenating {len(sbs_clips)} clips into {args.out}...")
    # Re-encode (not stream copy) when concatenating since clips might have
    # slightly different encoder params after tpad.
    concat_with_reencode(sbs_clips, args.out, fps=args.fps)

    if not args.keep_intermediate:
        for clip in sbs_clips:
            try:
                os.remove(clip)
            except OSError:
                pass

    print(f"\n✓ Done. Wrote {args.out}")


def concat_with_reencode(input_paths, out_path, fps):
    """Concat with re-encode (safer than -c copy when params differ)."""
    inputs = []
    for pth in input_paths:
        inputs += ["-i", pth]
    n = len(input_paths)
    filter_str = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1[outv]"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-r", str(fps),
        out_path,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()