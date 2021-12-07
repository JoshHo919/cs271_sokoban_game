import numpy as np
from copy import copy
from dataclasses import dataclass

INFEASIBLE = -1
SPACE = 0
ACTOR = 1
WALL = 2
BOX = 3
TARGET = 4
BOX_ON_TARGET = 5
ACTOR_ON_TARGET = 6

UP = np.array([-1, 0])
DOWN = np.array([1, 0])
LEFT = np.array([0, -1])
RIGHT = np.array([0, 1])
actions = {'UP': UP, 'LEFT': LEFT, 'DOWN': DOWN, 'RIGHT': RIGHT}

BASIC_REWARD = {'SPACE': -2, 'BOX_OFF_WALL': 0.6, 'BOX_BY_WALL': -1.1, 'INFEASIBLE': -199, \
                'ON_TARGET': 150, 'ON_SPACE': -1, 'OFF_TARGET': -200, 'DEADLOCK': -10e10, 'GOAL': 10e10}

class State:
    def __init__(self, map_array, actor, boxes, targets):
        self.map = np.copy(map_array)
        self.actor = np.copy(actor)
        self.boxes = np.copy(boxes)
        self.targets = np.copy(targets)
        self.key = state_hash(self)
        self.test = False

    @classmethod
    def from_config(cls, config_text):
        """
        Parses the content of input file and creates map representation.
        :param config_text: string from input file
        :return: a state instance with map and the locations of the objects on the map.
        """
        lines = config_text.split("\n")
        dimensions = lines[0].split(" ")
        wall_str = lines[1].split(" ")
        box_str = lines[2].split(" ")
        target_str = lines[3].split(" ")
        actor_str = lines[4].split(" ")

        np_map = np.zeros((int(dimensions[0]), int(dimensions[1])))
        np_actor = np.array([int(actor_str[0]) - 1, int(actor_str[1]) - 1])

        for i in range(1, int(wall_str[0]) + 1):
            np_map[int(wall_str[i * 2 - 1]) - 1, int(wall_str[i * 2]) - 1] = WALL

        boxes = []

        for i in range(1, int(box_str[0]) * 2 + 1, 2):
            np_map[int(box_str[i]) - 1, int(box_str[i + 1]) - 1] = BOX
            boxes.append([int(box_str[i]) - 1, int(box_str[i + 1]) - 1])
        np_boxes = np.array(boxes)

        targets = []
        for i in range(1, int(target_str[0]) * 2 + 1, 2):
            if np_map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] == BOX:
                np_map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] = BOX_ON_TARGET
            else:
                np_map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] = TARGET
            targets.append([int(target_str[i]) - 1, int(target_str[i + 1]) - 1])
        np_targets = np.array(targets)

        np_map[int(actor_str[0]) - 1, int(actor_str[1]) - 1] = ACTOR

        return cls(np_map, np_actor, np_boxes, np_targets)

    def __copy__(self):
        return State(np.copy(self.map), np.copy(self.actor), np.copy(self.boxes), np.copy(self.targets))

    def __eq__(self, other) :
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

# hash based on location of agent only
def loc_hash(state):
    r, c = state.map.shape
    b = max(r, c)
    return state.actor[0] * b + state.actor[1]

# return hash based on location of agent and boxes
def state_hash(state):
    sorted_boxes = np.array([]).reshape(0, 2)
    if len(state.boxes) > 0:
        sorted_boxes = state.boxes[np.lexsort((state.boxes[:, 1], state.boxes[:, 0]))]
    r, c = state.map.shape
    base = max(r, c)
    value = state.actor[0] + state.actor[1] * base
    for i, b in enumerate(sorted_boxes):
        j = 2 * (i + 1)
        value += b[0] * base ** j + b[1] * base ** (j + 1)
    return hash(value)

def step(state, action):
    new_state = copy(state)
    next_position = state.actor + actions[action]

    # conditions when pushing:
    # 1. wall: can't move
    # 2. space: move
    # 3. box -> space: push
    # 4. box -> box/wall: can't move
    # 5. box -> target: move onto target

    # update actor's current position for the new state depends on current actor on target or space
    if state.map[state.actor[0]][state.actor[1]] == ACTOR:
        new_state.map[state.actor[0]][state.actor[1]] = SPACE
    elif state.map[state.actor[0]][state.actor[1]] == ACTOR_ON_TARGET:
        new_state.map[state.actor[0]][state.actor[1]] = TARGET

    # if next actor position is space
    if state.map[next_position[0]][next_position[1]] == SPACE:
        new_state.map[next_position[0]][next_position[1]] = ACTOR
    # if next actor position is box
    elif state.map[next_position[0]][next_position[1]] in [BOX, BOX_ON_TARGET]:
        # the position box will be pushed on
        next_two_position = next_position + actions[action]

        new_state.map[next_position[0]][next_position[1]] = ACTOR \
            if state.map[next_position[0]][next_position[1]] == BOX else ACTOR_ON_TARGET

        new_state.map[next_two_position[0]][next_two_position[1]] = BOX if \
            state.map[next_two_position[0]][next_two_position[1]] == SPACE else BOX_ON_TARGET

        # modify box location been pushed
        for i in range(len(new_state.boxes)):
            if np.array_equal(new_state.boxes[i], next_position):
                new_state.boxes[i] = next_two_position
                break
        
    elif state.map[next_position[0]][next_position[1]] == TARGET:
        new_state.map[next_position[0]][next_position[1]] = ACTOR_ON_TARGET

    # update actor position in the new state
    new_state.actor = next_position
    new_state.key = state_hash(new_state)

    return new_state

def get_feasible_actions(state):
    feasible_actions = []
    for action in actions:
        if is_feasible_action(state, action):
            feasible_actions.append(action)
    return feasible_actions

def count_walls(state, position):
    count = 0
    for action in actions:
        next_position = position + actions[action]
        next_pos_status = get_location_status(state, next_position)
        if next_pos_status in [WALL, BOX_ON_TARGET, BOX]:
            count += 1
    return count

def get_reward(state, action, new_state):
    reward = 0

    if is_feasible_action(state, action):
        next_position = state.actor + actions[action]
        next_pos_status = get_location_status(state, next_position)
        if next_pos_status in [SPACE, TARGET]:
            reward = BASIC_REWARD['SPACE']
        elif next_pos_status in [BOX, BOX_ON_TARGET]:
            box_next_position = next_position + actions[action]
            if not is_out_of_bounds(state, box_next_position):
                box_next_pos_status = get_location_status(state, box_next_position)
                # push box off target
                if next_pos_status == BOX_ON_TARGET:
                    if box_next_pos_status == SPACE:
                        reward += BASIC_REWARD['OFF_TARGET']
                    elif box_next_pos_status == TARGET:
                        new_state.test = True
                        reward += BASIC_REWARD['ON_TARGET'] ** 2
                        wall_count = count_walls(state, box_next_position)
                        if wall_count >= 2:
                            reward *= (wall_count - 1) * 1000
                        box_next_position = box_next_position + actions[action]
                        while not is_out_of_bounds(state, box_next_position):
                            box_next_pos_status = get_location_status(state, box_next_position)
                            if box_next_pos_status == TARGET:
                                reward *= 5
                            else:
                                break;
                            box_next_position = box_next_position + actions[action]
                elif next_pos_status == BOX:
                    # infeasible push

                    if box_next_pos_status in [WALL, BOX, BOX_ON_TARGET]:
                        reward += BASIC_REWARD['INFEASIBLE']
                    elif box_next_pos_status == TARGET:
                        reward += BASIC_REWARD['ON_TARGET']

                        box_next_position = box_next_position + actions[action]
                        while not is_out_of_bounds(state, box_next_position):
                            box_next_pos_status = get_location_status(state, box_next_position)
                            if box_next_pos_status == TARGET:
                                reward += BASIC_REWARD['ON_TARGET'] * 2
                            else:
                                break
                            box_next_position = box_next_position + actions[action]
                    elif box_next_pos_status == SPACE:
                        reward += BASIC_REWARD['ON_SPACE']
                        current_pos_walls = count_walls(state,next_position)
                        next_pos_walls = count_walls(state,box_next_position) - 1
                        wall_diff = next_pos_walls - current_pos_walls
                        if wall_diff >= 0:
                            reward += BASIC_REWARD['BOX_BY_WALL'] ** (wall_diff + 1)
                        else:
                            reward += BASIC_REWARD['BOX_OFF_WALL'] * wall_diff
            else:
                reward += BASIC_REWARD['INFEASIBLE']
    else:
        reward += BASIC_REWARD['INFEASIBLE']

    if is_goal(new_state):
        reward += BASIC_REWARD['GOAL']
    elif is_deadlock(new_state, action):
        reward += BASIC_REWARD['DEADLOCK']

    return reward

def get_location_status(state, loc):
    if is_out_of_bounds(state, loc):
        return INFEASIBLE
    return state.map[loc[0], loc[1]]

# returns true if action is feasible in state, false otherwise
def is_feasible_action(state, action):
    next_position = state.actor + actions[action]
    if not is_out_of_bounds(state, next_position):
        if get_location_status(state, next_position) in [SPACE, TARGET]:
            return True
        next_two_position = next_position + actions[action]
        if not is_out_of_bounds(state, next_two_position):
            if (get_location_status(state, next_position) in [BOX, BOX_ON_TARGET] and \
                get_location_status(state, next_two_position) not in [BOX, WALL, BOX_ON_TARGET]):
                return True
    return False

def is_goal(state):
    count = 0
    for b in state.boxes:
        if get_location_status(state, b) == BOX_ON_TARGET:
            count += 1
    return count == len(state.boxes)

# state: current state
# action: action used to get to current state 
def is_deadlock(state, action):
    for loc in state.boxes:
        if is_immovable(state, loc) and get_location_status(state, loc) != BOX_ON_TARGET:
            return True

    loc = state.actor + actions[action]
    loc2 = loc + actions[action]
    if get_location_status(state, loc) == BOX and get_location_status(state, loc2) in [WALL, INFEASIBLE]:
        for dir, perp in [(['UP', 'DOWN'], ['LEFT', 'RIGHT']), (['LEFT', 'RIGHT'], ['UP', 'DOWN'])]:
            if action in dir:
                cts = True
                target_count = 0
                box_count = 0
                
                for a in perp:
                    l = loc + actions[a]
                    while get_location_status(state, l) not in [WALL, INFEASIBLE]:
                        if get_location_status(state, l) == TARGET:
                            target_count += 1
                        elif get_location_status(state, l) == BOX:
                            box_count += 1
                        j1 = get_location_status(state, l + actions[dir[0]])
                        j2 = get_location_status(state, l + actions[dir[1]])
                        if j1 not in [WALL, INFEASIBLE] and j2 not in [WALL, INFEASIBLE]:
                            cts = False
                        l += actions[a]
                if cts and box_count + 1 > target_count:
                    return True
    return False

def is_out_of_bounds(state, loc):
    r, c = state.map.shape
    return (loc[0] < 0 or loc[0] >= r) or (loc[1] < 0 or loc[1] >= c)


# returns true if a box at loc is immovable
def is_immovable(state, loc):
    for a1, a2 in [('UP', 'RIGHT'), ('RIGHT', 'DOWN'), ('DOWN', 'LEFT'), ('LEFT', 'UP')]:
        n1 = loc + actions[a1]
        n2 = loc + actions[a2]
        n1_status = get_location_status(state, n1)
        n2_status = get_location_status(state, n2)

        # are n1, n2 occupied by wall?
        o1 = n1_status in [WALL, INFEASIBLE, BOX, BOX_ON_TARGET]
        o2 = n2_status in [WALL, INFEASIBLE, BOX, BOX_ON_TARGET]
        if o1 and o2:
            if n1_status in [WALL, INFEASIBLE] and n2_status in [WALL, INFEASIBLE]:
                return True
            else:
                for n, status, a in [(n1, n1_status, a1), (n2, n2_status, a2)]:
                    if status in [BOX, BOX_ON_TARGET]:
                        for dir1, dir2 in [(['UP', 'DOWN'], ['LEFT', 'RIGHT']), (['LEFT', 'RIGHT'], ['UP', 'DOWN'])]:
                            # detect 2 box placement
                            if a in dir1:
                                l1 = [loc + actions[x] for x in dir2]
                                l2 = [n + actions[x] for x in dir2]
                                loc_blocked = any([get_location_status(state, x) in [WALL, INFEASIBLE, BOX, BOX_ON_TARGET] for x in l1])
                                n_blocked = any([get_location_status(state, x) in [WALL, INFEASIBLE, BOX, BOX_ON_TARGET] for x in l2])
                                if loc_blocked and n_blocked:
                                    return True
    return False