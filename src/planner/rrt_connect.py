import numpy as np
import random
from src.planner.collision_detector import JOINT_LIMITS, in_collision
from nn_collision_detector import NeuralCollisionChecker


checker = NeuralCollisionChecker(
    scene_point_cloud=np.random.default_rng(0).uniform(-1, 1, (2048, 3)).astype(np.float32),
)

def is_in_collision(q: list) -> bool:
    return checker.in_collision(q)

# ── batch-check a whole interpolated edge at once (recommended) ─
def edge_in_collision(q_start: np.ndarray, q_end: np.ndarray, n_steps: int) -> bool:
    waypoints = np.linspace(q_start, q_end, n_steps) 
    return checker.batch_in_collision(waypoints).any()

def farthest_free(q_start: np.ndarray, q_end: np.ndarray, n_steps: int) -> bool:
    waypoints = np.linspace(q_start, q_end, n_steps)
    collisions = checker.batch_in_collision(waypoints)
    if not collisions.any():
        return q_end
    position = collisions.argmax()
    return waypoints[position - 1]


# Step size of linear interpolation during collision checking
STEP_SIZE = 0.05

# Node for RRT Conenct planning
class RRT_Node:
    def __init__(self, conf):
        self.conf = np.array(conf)
        self.parent = None
        self.children = []

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)


def sample_conf() -> RRT_Node:
    sample = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS]
    sample_node = RRT_Node(sample)
    return sample_node
   
def find_nearest(rand_node: RRT_Node, node_list: list[RRT_Node]) -> RRT_Node:
    dist = [np.linalg.norm(rand_node.conf - node.conf) for node in node_list]
    return node_list[np.argmin(dist)]

def steer_to(rand_node: RRT_Node, nearest_node: RRT_Node, use_nn_collision: bool) -> bool:
    direction = rand_node.conf - nearest_node.conf
    dist = np.linalg.norm(direction)
    if use_nn_collision:
        return not edge_in_collision(nearest_node.conf, rand_node.conf, dist/STEP_SIZE)
    for i in np.arange(STEP_SIZE, dist, STEP_SIZE):
        temp_conf = nearest_node.conf + direction * i / dist
        if in_collision(temp_conf):
            return False
    return not in_collision(rand_node.conf)

def steer_to_until(rand_node: RRT_Node, nearest_node: RRT_Node, use_nn_collision: bool) -> RRT_Node:
    direction = rand_node.conf - nearest_node.conf
    dist = np.linalg.norm(direction)
    if use_nn_collision:
        return farthest_free(nearest_node.conf, rand_node.conf, dist/STEP_SIZE)
    prev = nearest_node.conf
    for i in np.arange(STEP_SIZE, dist, STEP_SIZE):
        temp_conf = nearest_node.conf + direction * i / dist
        if in_collision(temp_conf):
            return RRT_Node(prev)
        prev = temp_conf
    if in_collision(rand_node.conf):
        return RRT_Node(prev)
    return rand_node

def path_smoothing(path: list[np.ndarray]) -> list[np.ndarray]:
    '''
    Performs path smoothing

    Args:
        path (list[np.ndarray]): The path to perform smoothing on. 

    Returns:
        list[np.ndarray]: The path after smoothing.
    '''
    # number of iterations to attempt smoothing
    N = 100

    for _ in range(N):
        if len(path) <= 3:
            break
        [one, two] = np.random.choice(len(path), 2)
        while abs(one - two) < 2:
            [one, two] = np.random.choice(len(path), 2)
        # print((int(one), int(two)))
        if steer_to(RRT_Node(path[one]), RRT_Node(path[two])):
            path = path[:min(one, two) + 1] + path[max(one, two):]
        # if len(path) == 3:
        #     break
    return path

def RRTConnect(start_conf, goal_conf, use_nn_collision) -> list[np.ndarray]:
    start_node = RRT_Node(start_conf)
    goal_node = RRT_Node(goal_conf)
    T1: list[RRT_Node] = [start_node]
    T2: list[RRT_Node] = [goal_node]

    while True:
        # Sample q_rand
        q_rand = sample_conf()
        # Find q_nearest
        q_nearest = find_nearest(q_rand, T1)
        # if Steer from q_nearest to q_rand is possible (collision check)
        if steer_to(q_rand, q_nearest, use_nn_collision):
            # then add q_rand to tree and list of nodes
            q_nearest.children.append(q_rand)
            q_rand.parent = q_nearest
            T1.append(q_rand)
        else:
            continue

        # find nearest for other tree
        q_near_goal = find_nearest(q_rand, T2)
        q_furthest = steer_to_until(q_rand, q_near_goal)
        # if can connect then return path
        # otherwise swap the trees
        if np.array_equal(q_rand.conf, q_furthest.conf):
            path1 = []
            cur = q_rand
            while cur:
                path1.append(cur.conf)
                cur = cur.parent
            path2 = []
            cur = q_near_goal
            while cur:
                path2.append(cur.conf)
                cur = cur.parent
            
            if np.array_equal(path1[-1], start_node.conf):
                return path1[::-1] + path2
            return path2[::-1] + path1
        else:
            temp_tree = T1
            T1 = T2
            T2 = temp_tree
