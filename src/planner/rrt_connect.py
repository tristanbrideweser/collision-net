import numpy as np
import random
from src.planner.collision_detector import JOINT_LIMITS

# Constants for the RRT logic
STEP_SIZE = 0.05
MAX_ITER = 1000

class RRT_Node:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None

def sample_conf() -> RRT_Node:
    """Samples a random configuration within joint limits."""
    sample = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS]
    return RRT_Node(sample)
   
def find_nearest(target_conf: np.ndarray, node_list: list[RRT_Node]) -> RRT_Node:
    """Finds the closest node in the tree to the target configuration."""
    dist = [np.linalg.norm(target_conf - node.conf) for node in node_list]
    return node_list[np.argmin(dist)]

def edge_is_free(q_start: np.ndarray, q_end: np.ndarray, collision_fn, n_steps=15) -> bool:
    """Checks if the path between two nodes is collision-free."""
    # Interpolate waypoints along the edge
    waypoints = np.linspace(q_start, q_end, n_steps)
    # Check each waypoint. If using Neural, this can be batch-processed.
    for q in waypoints:
        if collision_fn(q):
            return False
    return True

def steer_to_until(rand_conf: np.ndarray, nearest_node: np.ndarray, collision_fn) -> np.ndarray:
    """Steers from nearest toward rand until a collision is hit."""
    direction = rand_conf - nearest_node
    dist = np.linalg.norm(direction)
    steps = int(dist / STEP_SIZE)
    if steps < 1: return rand_conf
    
    prev = nearest_node
    for i in range(1, steps + 1):
        temp_conf = nearest_node + direction * (i / steps)
        if collision_fn(temp_conf):
            return prev
        prev = temp_conf
    return rand_conf

def reconstruct_path(node_a: RRT_Node, node_b: RRT_Node, swapped: bool) -> list[np.ndarray]:
    """Reconstructs the full path from two trees."""
    path_a = []
    curr = node_a
    while curr:
        path_a.append(curr.conf)
        curr = curr.parent
    path_a.reverse()
    
    path_b = []
    curr = node_b
    while curr:
        path_b.append(curr.conf)
        curr = curr.parent
        
    if swapped:
        return path_b[::-1] + path_a
    return path_a + path_b

def rrt_connect(start_conf, goal_conf, collision_fn) -> list[np.ndarray]:
    """
    Standard RRT-Connect implementation that accepts a collision function.
    This allows for direct comparison between Geometric and Neural checkers.
    """
    if collision_fn(start_conf) or collision_fn(goal_conf):
        return None
    
    T_a = [RRT_Node(start_conf)]
    T_b = [RRT_Node(goal_conf)]
    swapped = False

    for _ in range(MAX_ITER):
        # 1. Extend Tree A
        q_rand = sample_conf().conf
        q_near_a = find_nearest(q_rand, T_a)
        
        # Check if the sampled direction is free
        if edge_is_free(q_near_a.conf, q_rand, collision_fn):
            q_new_a = RRT_Node(q_rand)
            q_new_a.parent = q_near_a
            T_a.append(q_new_a)
            
            # 2. Try to connect Tree B to the new node in Tree A
            q_near_b = find_nearest(q_new_a.conf, T_b)
            q_target_b = steer_to_until(q_new_a.conf, q_near_b.conf, collision_fn)
            
            q_new_b = RRT_Node(q_target_b)
            q_new_b.parent = q_near_b
            T_b.append(q_new_b)
            
            # 3. Check if trees have met
            if np.linalg.norm(q_new_b.conf - q_new_a.conf) < 1e-3:
                return reconstruct_path(q_new_a, q_new_b, swapped)

        # 4. Swap trees for balanced growth
        T_a, T_b = T_b, T_a
        swapped = not swapped
    
    return None