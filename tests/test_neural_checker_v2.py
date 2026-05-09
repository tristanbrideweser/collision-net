"""
test_neural_checker_v2.py — Test neural collision checker against pre-generated labels.

Uses the train/val/test.npz files which have ground truth labels from geometric checker.

Usage:
    cd src/planner
    python test_neural_checker_v2.py --model ../model/checkpoints/best.pt --data-dir ../data/scenes
    
    # Test on specific split
    python test_neural_checker_v2.py --model ../model/checkpoints/best.pt --data-dir ../data/scenes --split test
    
    # Limit number of samples
    python test_neural_checker_v2.py --model ../model/checkpoints/best.pt --data-dir ../data/scenes --n-samples 1000
"""

import argparse
import os
import sys
import time
import numpy as np

# Add src/ to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from planner.nn_collision_detector import NeuralCollisionChecker


def load_split_data(data_dir, split="test"):
    """Load configs, labels, and scene_ids from a split file."""
    split_file = os.path.join(data_dir, f"{split}.npz")
    data = np.load(split_file)
    return data["configs"], data["labels"], data["scene_ids"]


def load_scene_point_cloud(data_dir, scene_id):
    """Load point cloud for a specific scene."""
    scene_file = os.path.join(data_dir, "scenes", f"scene_{scene_id:04d}.npz")
    return np.load(scene_file)["point_cloud"]


def main():
    parser = argparse.ArgumentParser(description="Test neural checker against ground truth labels")
    parser.add_argument("--model", type=str, default="../model/checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="../data/scenes")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=None, help="Limit samples (default: all)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading {args.split} split from {args.data_dir}...")
    configs, labels, scene_ids = load_split_data(args.data_dir, args.split)
    
    # Subsample if requested
    if args.n_samples and args.n_samples < len(labels):
        idx = np.random.choice(len(labels), args.n_samples, replace=False)
        configs = configs[idx]
        labels = labels[idx]
        scene_ids = scene_ids[idx]
    
    print(f"Samples: {len(labels)}")
    print(f"Ground truth collision rate: {labels.mean()*100:.1f}%")
    
    # Initialize checker with dummy point cloud
    dummy_pc = np.zeros((2048, 3), dtype=np.float32)
    checker = NeuralCollisionChecker(
        scene_point_cloud=dummy_pc,
        model_path=args.model,
        threshold=args.threshold,
    )
    
    # Group by scene for efficient batching
    unique_scenes = np.unique(scene_ids)
    print(f"Unique scenes: {len(unique_scenes)}")
    
    all_preds = np.zeros(len(labels), dtype=bool)
    all_probs = np.zeros(len(labels), dtype=np.float32)
    
    total_neural_time = 0
    
    for scene_id in unique_scenes:
        mask = scene_ids == scene_id
        scene_configs = configs[mask]
        
        # Load and set scene point cloud
        pc = load_scene_point_cloud(args.data_dir, scene_id)
        checker.update_scene(pc)
        
        # Batch prediction
        t0 = time.time()
        preds = checker.batch_in_collision(scene_configs)
        probs = checker.batch_collision_probabilities(scene_configs)
        total_neural_time += time.time() - t0
        
        all_preds[mask] = preds
        all_probs[mask] = probs
    
    # Compute metrics
    gt = labels.astype(bool)
    tp = np.sum(all_preds & gt)
    tn = np.sum(~all_preds & ~gt)
    fp = np.sum(all_preds & ~gt)
    fn = np.sum(~all_preds & gt)
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS ({args.split} split)")
    print(f"{'='*60}")
    print(f"Samples tested: {len(labels)}")
    print(f"Threshold: {args.threshold}")
    print(f"\nGround truth collision rate: {gt.mean()*100:.1f}%")
    print(f"Predicted collision rate:    {all_preds.mean()*100:.1f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Free    Collision")
    print(f"  Actual Free   {tn:5d}     {fp:5d}")
    print(f"  Actual Col    {fn:5d}     {tp:5d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:       {accuracy*100:.2f}%")
    print(f"  Precision:      {precision*100:.2f}%")
    print(f"  Recall:         {recall*100:.2f}%")
    print(f"  F1 Score:       {f1*100:.2f}%")
    print(f"  False Neg Rate: {fnr*100:.2f}% {'⚠️  HIGH - missed collisions!' if fnr > 0.05 else '✓'}")
    print(f"  False Pos Rate: {fpr*100:.2f}% (conservative errors, generally safe)")
    
    print(f"\nSpeed:")
    print(f"  Total neural time: {total_neural_time*1000:.1f} ms")
    print(f"  Per-sample:        {total_neural_time/len(labels)*1000:.3f} ms")
    print(f"  Throughput:        {len(labels)/total_neural_time:,.0f} configs/sec")
    
    # Probability distribution analysis
    print(f"\nProbability Analysis:")
    print(f"  Mean prob (actual free):      {all_probs[~gt].mean():.3f}")
    print(f"  Mean prob (actual collision): {all_probs[gt].mean():.3f}")
    
    # Threshold sweep
    print(f"\nThreshold Analysis:")
    print(f"  {'Thresh':>6}  {'Acc':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'FNR':>6}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = all_probs >= thresh
        tp_t = np.sum(preds_t & gt)
        tn_t = np.sum(~preds_t & ~gt)
        fp_t = np.sum(preds_t & ~gt)
        fn_t = np.sum(~preds_t & gt)
        acc_t = (tp_t + tn_t) / len(labels)
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0
        marker = " <--" if thresh == args.threshold else ""
        print(f"  {thresh:>6.2f}  {acc_t*100:>5.1f}%  {prec_t*100:>5.1f}%  {rec_t*100:>5.1f}%  {f1_t*100:>5.1f}%  {fnr_t*100:>5.1f}%{marker}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()