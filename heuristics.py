import environment
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from queue import Queue

class Node:
    def __init__(self, state, prev, cost):
        self.state = state
        self.prev = prev
        self.cost = cost

def get_distance_table(state):
    table = {}
    s = deepcopy(state)
    r, c = s.map.shape

    # delete boxes and targets
    for l in s.boxes:
        s.map[l[0]][l[1]] = environment.SPACE
    for l in s.targets:
        s.map[l[0]][l[1]] = environment.SPACE
    s.boxes = np.array([]).reshape(0, 0)
    s.targets = np.array([]).reshape(0, 0)

    for i in range(r):
        for j in range(c):
            if int(s.map[i, j]) != environment.WALL:
                new_state = deepcopy(s)
                
                # move actor to set location
                loc = new_state.actor
                new_state.map[loc[0], loc[1]] = environment.SPACE
                new_state.map[i, j] = environment.ACTOR
                new_state.actor = np.array([i, j])
                
                # compute distance matrix for location
                compute_distance_matrix(table, new_state)
    return table

def compute_distance_matrix(table, state):
    r, c = state.map.shape
    b = max(r, c)
    distance_matrix = np.full((r, c), -1)
    explored = {}
    q = Queue()
    q.put(Node(state, None, 0))

    curr_loc = state.actor
    loc_hash = b * curr_loc[0] + curr_loc[1]

    # use bfs to find shortest distance for each pair of points
    while not q.empty():
        node = q.get()
        hash_val = environment.loc_hash(node.state)
        if hash_val not in explored:
            explored[hash_val] = 1
            loc = node.state.actor
            actions = environment.get_feasible_actions(node.state)
            for a in actions:
                new_state = environment.step(node.state, a)
                new_node = Node(new_state, node, node.cost + 1)
                if environment.loc_hash(new_state) not in explored:
                    q.put(new_node)
            distance_matrix[loc[0], loc[1]] = node.cost + 1
    
    # save distance matrix
    table[loc_hash] = distance_matrix

# used for EMM heuristic
class MinMatcher:
    def __init__(self, distance_table):
        self.distance_table = distance_table

    def get_min_matching_cost(self, state):
        row, col = state.map.shape
        base = max(row, col)
        
        # create dist_matrix for given state (maps min distance of boxes to targets)
        dist_matrix = np.zeros((len(state.boxes), len(state.targets)))
        for i, b in enumerate(state.boxes):
            for j, t in enumerate(state.targets):
                dist_matrix[i, j] = self.distance_table[t[0] * base + t[1]][b[0]][b[1]]
        
        # compute minimum perfect matching
        r, c = linear_sum_assignment(dist_matrix)

        # compute the cost of the matching
        cost = 0
        for i, j in zip(r, c):
            cost += dist_matrix[i, j]
        return cost

class Heuristic:
    def __init__(self, dist_table):
        self.dist_table = dist_table # pairwise min distances

    def heuristic(self, state):
        pass

class EMMHeuristic(Heuristic):
    def __init__(self, dist_table):
        super().__init__(dist_table)
        self.min_matcher = MinMatcher(dist_table)

    def heuristic(self, state):
        return self.min_matcher.get_min_matching_cost(state)

class AgentBoxHeuristic(Heuristic):
    def __init__(self, dist_table):
        super().__init__(dist_table)

    def heuristic(self, state):
        m = 10e10
        dist = self.dist_table[environment.loc_hash(state)]

        for b in state.boxes:
            if state.map[b[0], b[1]] != environment.BOX_ON_TARGET:
                d = dist[b[0], b[1]] # min distance to box location from actor
                m = min(m, d)
        return 0 if m == 10e10 else m