import numpy as np
import random
from src.planner.collision_detector import JOINT_LIMITS, in_collision

STEP_SIZE = 0.05

class RRT_Node:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None

def sample_conf() -> RRT_Node:
    sample = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS]
    return RRT_Node(sample)
   
def find_nearest(target_conf: np.ndarray, node_list: list[RRT_Node]) -> RRT_Node:
    dist = [np.linalg.norm(target_conf - node.conf) for node in node_list]
    return node_list[np.argmin(dist)]

def steer_to_until(rand_conf: np.ndarray, nearest_node: RRT_Node) -> RRT_Node:
    """Extends from nearest_node toward rand_conf until collision or target reached."""
    direction = rand_conf - nearest_node.conf
    dist = np.linalg.norm(direction)
    if dist < 1e-6: return nearest_node
    
    unit_dir = direction / dist
    # Determine how many steps to take
    steps = np.arange(STEP_SIZE, dist, STEP_SIZE)
    last_valid_conf = nearest_node.conf
    
    for s in steps:
        temp_conf = nearest_node.conf + unit_dir * s
        if in_collision(temp_conf):
            return RRT_Node(last_valid_conf) if not np.array_equal(last_valid_conf, nearest_node.conf) else None
        last_valid_conf = temp_conf
    
    # Check final target if the loop finishes
    if not in_collision(rand_conf):
        return RRT_Node(rand_conf)
    return RRT_Node(last_valid_conf)

def RRTConnect(start_conf, goal_conf, max_iter=5000) -> list[np.ndarray]:
    start_node = RRT_Node(start_conf)
    goal_node = RRT_Node(goal_conf)
    
    # Trees: T_a starts at 'start', T_b starts at 'goal'
    T_a = [start_node]
    T_b = [goal_node]
    
    # We need to keep track of which tree is which for path reconstruction
    swapped = False

    if in_collision(start_conf) or in_collision(goal_conf):
        print("Error: Start or Goal is in collision!")
        return None
    
    for i in range(max_iter):
        # 1. Sample and extend T_a
        q_rand = sample_conf()
        q_near_a = find_nearest(q_rand.conf, T_a)
        q_new_a = steer_to_until(q_rand.conf, q_near_a)
        
        if q_new_a:
            q_new_a.parent = q_near_a
            T_a.append(q_new_a)
            
            # 2. Try to connect T_b to the node we just added to T_a
            q_near_b = find_nearest(q_new_a.conf, T_b)
            q_new_b = steer_to_until(q_new_a.conf, q_near_b)
            
            if q_new_b:
                q_new_b.parent = q_near_b
                T_b.append(q_new_b)
                
                # 3. Check if they met (Connection Successful)
                if np.linalg.norm(q_new_b.conf - q_new_a.conf) < 1e-3:
                    return reconstruct_full_path(q_new_a, q_new_b, swapped)

        # 4. Swap trees so both grow equally
        T_a, T_b = T_b, T_a
        swapped = not swapped
    
    print("Planner failed to find path within max iterations.")
    return None

def reconstruct_full_path(node_a, node_b, swapped):
    # Path from T_a root to connection point
    path_a = []
    curr = node_a
    while curr:
        path_a.append(curr.conf)
        curr = curr.parent
    path_a.reverse()
    
    # Path from T_b connection point to root
    path_b = []
    curr = node_b
    while curr:
        path_b.append(curr.conf)
        curr = curr.parent
        
    if swapped:
        # T_a is goal-rooted, T_b is start-rooted
        return path_b[::-1] + path_a
    else:
        # T_a is start-rooted, T_b is goal-rooted
        return path_a + path_b
