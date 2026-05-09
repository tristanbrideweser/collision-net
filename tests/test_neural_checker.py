"""
test_neural_checker.py — Compare neural collision checker against geometric ground truth.

Usage:
    cd src/planner
    python test_neural_checker.py --model ../model/checkpoints/best.pt --scene ../data/scenes/scenes/scene_0000.npz
    
    # Test on multiple scenes
    python test_neural_checker.py --model ../model/checkpoints/best.pt --scenes-dir ../data/scenes/scenes --n-scenes 5
    
    # More configs for better stats
    python test_neural_checker.py --model ../model/checkpoints/best.pt --scene ../data/scenes/scenes/scene_0000.npz --n-configs 500
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


def sample_random_configs(n):
    """Sample random configurations within joint limits."""
    return np.random.uniform(JOINT_LOWER, JOINT_UPPER, (n, 7))


def spawn_obstacles_from_pointcloud(point_cloud, voxel_size=0.05):
    """
    Approximate obstacle spawning from point cloud for geometric checking.
    This is a rough approximation - for accurate testing, load the actual scene.
    """
    # For proper testing, we'd need the original obstacle definitions
    # This creates small spheres at point cloud locations as a rough proxy
    obj_ids = []
    # Subsample points to avoid too many bodies
    n_spheres = min(50, len(point_cloud))
    indices = np.random.choice(len(point_cloud), n_spheres, replace=False)
    
    for idx in indices:
        pos = point_cloud[idx].tolist()
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=voxel_size)
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=pos)
        obj_ids.append(body_id)
    
    return obj_ids


def test_on_scene(neural_checker, scene_path, n_configs=100, panda_id=None, verbose=True):
    """Test neural checker against geometric checker on a single scene."""
    
    # Load scene
    scene_data = np.load(scene_path)
    point_cloud = scene_data["point_cloud"]
    scene_type = str(scene_data.get("scene_type", "unknown"))
    
    # Update neural checker with this scene
    neural_checker.update_scene(point_cloud)
    
    # Sample random configs
    configs = sample_random_configs(n_configs)
    
    # Neural predictions (batched - fast)
    t0 = time.time()
    neural_preds = neural_checker.batch_in_collision(configs)
    neural_time = time.time() - t0
    
    # Geometric ground truth (slow)
    t0 = time.time()
    geometric_preds = np.array([geometric_in_collision(c.tolist()) for c in configs])
    geometric_time = time.time() - t0
    
    # Compute metrics
    tp = np.sum(neural_preds & geometric_preds)      # True positives
    tn = np.sum(~neural_preds & ~geometric_preds)    # True negatives
    fp = np.sum(neural_preds & ~geometric_preds)     # False positives (safe errors)
    fn = np.sum(~neural_preds & geometric_preds)     # False negatives (dangerous!)
    
    accuracy = (tp + tn) / n_configs
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # False negative rate is critical for safety
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    speedup = geometric_time / neural_time if neural_time > 0 else float('inf')
    
    results = {
        "scene": os.path.basename(scene_path),
        "scene_type": scene_type,
        "n_configs": n_configs,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fnr": fnr,
        "neural_time_ms": neural_time * 1000,
        "geometric_time_ms": geometric_time * 1000,
        "speedup": speedup,
        "collision_rate_gt": geometric_preds.mean(),
        "collision_rate_pred": neural_preds.mean(),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scene: {results['scene']} ({scene_type})")
        print(f"{'='*60}")
        print(f"Configs tested: {n_configs}")
        print(f"Ground truth collision rate: {results['collision_rate_gt']*100:.1f}%")
        print(f"Predicted collision rate:    {results['collision_rate_pred']*100:.1f}%")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TN: {tn:4d}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy*100:.1f}%")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall:    {recall*100:.1f}%")
        print(f"  F1 Score:  {f1*100:.1f}%")
        print(f"  FN Rate:   {fnr*100:.1f}% {'⚠️  HIGH' if fnr > 0.05 else '✓'}")
        print(f"\nSpeed:")
        print(f"  Neural:    {results['neural_time_ms']:.1f} ms")
        print(f"  Geometric: {results['geometric_time_ms']:.1f} ms")
        print(f"  Speedup:   {speedup:.1f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test neural collision checker")
    parser.add_argument("--model", type=str, default="../model/checkpoints/best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to single scene .npz file")
    parser.add_argument("--scenes-dir", type=str, default=None,
                        help="Path to directory of scene .npz files")
    parser.add_argument("--n-scenes", type=int, default=5,
                        help="Number of scenes to test (if using --scenes-dir)")
    parser.add_argument("--n-configs", type=int, default=200,
                        help="Number of configs to test per scene")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Collision probability threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Setup PyBullet (headless)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    # Initialize geometric collision checker with empty obstacles
    # (we'll update per-scene using point cloud approximation)
    init_collision_checker(panda_id=panda_id, obstacle_ids=[], table_id=None)
    
    # Load neural checker
    # Need a dummy point cloud to initialize
    dummy_pc = np.random.randn(2048, 3).astype(np.float32)
    neural_checker = NeuralCollisionChecker(
        scene_point_cloud=dummy_pc,
        model_path=args.model,
        threshold=args.threshold,
    )
    
    # Collect scenes to test
    if args.scene:
        scene_files = [args.scene]
    elif args.scenes_dir:
        scene_files = sorted(glob.glob(os.path.join(args.scenes_dir, "scene_*.npz")))
        scene_files = scene_files[:args.n_scenes]
    else:
        # Default: look for scenes in standard location
        default_dir = os.path.join(SRC_DIR, "data", "scenes", "scenes")
        if os.path.exists(default_dir):
            scene_files = sorted(glob.glob(os.path.join(default_dir, "scene_*.npz")))
            scene_files = scene_files[:args.n_scenes]
        else:
            print(f"No scenes found. Specify --scene or --scenes-dir")
            return
    
    if not scene_files:
        print("No scene files found!")
        return
    
    print(f"Testing {len(scene_files)} scene(s) with {args.n_configs} configs each")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    
    # Test each scene
    all_results = []
    for scene_file in scene_files:
        results = test_on_scene(
            neural_checker=neural_checker,
            scene_path=scene_file,
            n_configs=args.n_configs,
            panda_id=panda_id,
            verbose=True,
        )
        all_results.append(results)
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL SCENES")
        print(f"{'='*60}")
        
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        avg_f1 = np.mean([r["f1"] for r in all_results])
        avg_fnr = np.mean([r["fnr"] for r in all_results])
        avg_speedup = np.mean([r["speedup"] for r in all_results])
        total_fn = sum(r["fn"] for r in all_results)
        total_configs = sum(r["n_configs"] for r in all_results)
        
        print(f"Average Accuracy:  {avg_acc*100:.1f}%")
        print(f"Average F1:        {avg_f1*100:.1f}%")
        print(f"Average FN Rate:   {avg_fnr*100:.1f}%")
        print(f"Total FN:          {total_fn}/{total_configs}")
        print(f"Average Speedup:   {avg_speedup:.1f}x")
    
    p.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()