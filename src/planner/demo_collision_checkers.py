"""
demo_collision_checkers.py — Visual comparison of neural vs geometric collision checking.

Shows a PyBullet GUI with the robot moving through random configurations,
coloring the robot based on collision predictions:
  - Green: Both agree FREE
  - Red:   Both agree COLLISION  
  - Yellow: Neural says COLLISION, Geometric says FREE (false positive - safe)
  - Purple: Neural says FREE, Geometric says COLLISION (false negative - dangerous!)

Usage:
    cd src/planner
    python demo_collision_checkers.py --model ../model/checkpoints/best.pt --scene ../data/scenes/scenes/scene_0000.npz
    
    # Slower animation
    python demo_collision_checkers.py --model ../model/checkpoints/best.pt --scene ../data/scenes/scenes/scene_0000.npz --delay 1.0
    
    # Cycle through multiple scenes
    python demo_collision_checkers.py --model ../model/checkpoints/best.pt --scenes-dir ../data/scenes/scenes --n-scenes 5
"""

import argparse
import os
import sys
import time
import glob
import numpy as np
import pybullet as p
import pybullet_data

# Add src/ to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from planner.collision_detector import (
    init_collision_checker,
    in_collision as geometric_in_collision,
    JOINT_LIMITS,
)
from planner.nn_collision_detector import NeuralCollisionChecker

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS])
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS])

# Colors (RGBA)
COLOR_BOTH_FREE = [0.2, 0.8, 0.2, 1.0]       # Green - both say free
COLOR_BOTH_COLLISION = [0.8, 0.2, 0.2, 1.0]  # Red - both say collision
COLOR_FALSE_POSITIVE = [1.0, 0.8, 0.0, 1.0]  # Yellow - neural says collision, geometric says free
COLOR_FALSE_NEGATIVE = [0.6, 0.0, 0.8, 1.0]  # Purple - neural says free, geometric says collision (DANGER)
COLOR_OBSTACLE = [0.5, 0.5, 0.6, 0.8]


def set_robot_color(panda_id, color):
    """Set all robot links to a specific color."""
    for link_id in range(-1, 11):  # -1 is base, 0-10 are links
        p.changeVisualShape(panda_id, link_id, rgbaColor=color)


def set_robot_config(panda_id, config):
    """Set robot to a configuration."""
    for i in range(7):
        p.resetJointState(panda_id, i, config[i])


def sample_random_config():
    """Sample a random configuration."""
    return np.random.uniform(JOINT_LOWER, JOINT_UPPER)


def spawn_obstacles_from_pointcloud(point_cloud, subsample=100, radius=0.03):
    """Create visible obstacles from point cloud for geometric checking."""
    obj_ids = []
    n = min(subsample, len(point_cloud))
    indices = np.random.choice(len(point_cloud), n, replace=False)
    
    for idx in indices:
        pos = point_cloud[idx].tolist()
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=COLOR_OBSTACLE)
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=pos)
        obj_ids.append(body_id)
    
    return obj_ids


def clear_obstacles(obj_ids):
    """Remove obstacles from scene."""
    for obj_id in obj_ids:
        p.removeBody(obj_id)


def create_legend():
    """Add text overlay explaining colors."""
    # PyBullet doesn't have great text support, so we'll print to console
    print("\n" + "="*60)
    print("COLOR LEGEND:")
    print("  🟢 GREEN  - Both agree: FREE")
    print("  🔴 RED    - Both agree: COLLISION")
    print("  🟡 YELLOW - False Positive (Neural=collision, Geometric=free)")
    print("  🟣 PURPLE - False Negative (Neural=free, Geometric=collision) ⚠️")
    print("="*60 + "\n")


def run_demo(neural_checker, panda_id, obstacle_ids, n_configs=50, delay=0.5):
    """Run the visual demo for a scene."""
    
    stats = {"both_free": 0, "both_col": 0, "fp": 0, "fn": 0}
    
    for i in range(n_configs):
        config = sample_random_config()
        
        # Get predictions
        neural_pred = neural_checker.in_collision(config.tolist())
        neural_prob = neural_checker.collision_probability(config.tolist())
        geometric_pred = geometric_in_collision(config.tolist())
        
        # Determine color and category
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
        else:  # not neural_pred and geometric_pred
            color = COLOR_FALSE_NEGATIVE
            status = "FALSE NEG ⚠️"
            stats["fn"] += 1
        
        # Update visualization
        set_robot_config(panda_id, config)
        set_robot_color(panda_id, color)
        
        # Print status
        print(f"[{i+1:3d}/{n_configs}] {status:12s} | Neural: {neural_prob:.2f} | "
              f"Geo: {'COL' if geometric_pred else 'FREE'}")
        
        time.sleep(delay)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Visual demo of collision checkers")
    parser.add_argument("--model", type=str, default="../model/checkpoints/best.pt")
    parser.add_argument("--scene", type=str, default=None, help="Single scene .npz")
    parser.add_argument("--scenes-dir", type=str, default=None, help="Directory of scenes")
    parser.add_argument("--n-scenes", type=int, default=3, help="Number of scenes to demo")
    parser.add_argument("--n-configs", type=int, default=30, help="Configs per scene")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between configs (seconds)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--obstacle-radius", type=float, default=0.03)
    parser.add_argument("--n-obstacle-spheres", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Collect scene files
    if args.scene:
        scene_files = [args.scene]
    elif args.scenes_dir:
        scene_files = sorted(glob.glob(os.path.join(args.scenes_dir, "scene_*.npz")))
        scene_files = scene_files[:args.n_scenes]
    else:
        default_dir = os.path.join(SRC_DIR, "data", "scenes", "scenes")
        scene_files = sorted(glob.glob(os.path.join(default_dir, "scene_*.npz")))
        scene_files = scene_files[:args.n_scenes]
    
    if not scene_files:
        print("No scene files found!")
        return
    
    # Setup PyBullet GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setGravity(0, 0, -9.81)
    
    # Camera setup
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0.3, 0, 0.3]
    )
    
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    # Initialize neural checker with dummy point cloud
    dummy_pc = np.zeros((2048, 3), dtype=np.float32)
    neural_checker = NeuralCollisionChecker(
        scene_point_cloud=dummy_pc,
        model_path=args.model,
        threshold=args.threshold,
    )
    
    create_legend()
    
    print(f"Running demo on {len(scene_files)} scene(s)")
    print(f"Configs per scene: {args.n_configs}")
    print(f"Delay: {args.delay}s")
    print("\nPress Ctrl+C to exit\n")
    
    total_stats = {"both_free": 0, "both_col": 0, "fp": 0, "fn": 0}
    obstacle_ids = []
    
    try:
        for scene_idx, scene_file in enumerate(scene_files):
            # Clear previous obstacles
            if obstacle_ids:
                clear_obstacles(obstacle_ids)
            
            # Load scene
            scene_data = np.load(scene_file)
            point_cloud = scene_data["point_cloud"]
            scene_type = str(scene_data.get("scene_type", "unknown"))
            
            print(f"\n{'='*60}")
            print(f"Scene {scene_idx+1}/{len(scene_files)}: {os.path.basename(scene_file)}")
            print(f"Type: {scene_type}")
            print(f"{'='*60}")
            
            # Spawn obstacles for geometric checker
            obstacle_ids = spawn_obstacles_from_pointcloud(
                point_cloud, 
                subsample=args.n_obstacle_spheres,
                radius=args.obstacle_radius
            )
            
            # Initialize geometric checker
            init_collision_checker(panda_id=panda_id, obstacle_ids=obstacle_ids, table_id=None)
            
            # Update neural checker
            neural_checker.update_scene(point_cloud)
            
            # Run demo
            stats = run_demo(
                neural_checker=neural_checker,
                panda_id=panda_id,
                obstacle_ids=obstacle_ids,
                n_configs=args.n_configs,
                delay=args.delay,
            )
            
            # Accumulate stats
            for k in total_stats:
                total_stats[k] += stats[k]
            
            # Scene summary
            total = sum(stats.values())
            print(f"\nScene Summary:")
            print(f"  Both Free:       {stats['both_free']:3d} ({stats['both_free']/total*100:.1f}%)")
            print(f"  Both Collision:  {stats['both_col']:3d} ({stats['both_col']/total*100:.1f}%)")
            print(f"  False Positives: {stats['fp']:3d} ({stats['fp']/total*100:.1f}%)")
            print(f"  False Negatives: {stats['fn']:3d} ({stats['fn']/total*100:.1f}%) {'⚠️' if stats['fn'] > 0 else '✓'}")
            
            if scene_idx < len(scene_files) - 1:
                input("\nPress Enter for next scene...")
        
        # Final summary
        total = sum(total_stats.values())
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total configs tested: {total}")
        print(f"  Agreement:        {total_stats['both_free'] + total_stats['both_col']:3d} ({(total_stats['both_free'] + total_stats['both_col'])/total*100:.1f}%)")
        print(f"  False Positives:  {total_stats['fp']:3d} ({total_stats['fp']/total*100:.1f}%) - safe errors")
        print(f"  False Negatives:  {total_stats['fn']:3d} ({total_stats['fn']/total*100:.1f}%) - dangerous!")
        
        accuracy = (total_stats['both_free'] + total_stats['both_col']) / total
        print(f"\n  Accuracy: {accuracy*100:.1f}%")
        
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    
    p.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()