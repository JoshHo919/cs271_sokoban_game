import numpy as np
from copy import copy
from dataclasses import dataclass

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

BASIC_REWARD = {'SPACE': -20, 'BOX_BY_BOX': -4, 'BOX_BY_WALL': -2, 'INFEASIBLE': -9999,\
                'ON_TARGET': 30, 'ON_SPACE': -15, 'OFF_TARGET': -50, 'DEADLOCK': -10e10, 'GOAL': 10e10}

class State:
    def __init__(self, map_array, actor, boxes, targets):
        self.map = np.copy(map_array)
        self.actor = np.copy(actor)
        self.boxes = np.copy(boxes)
        self.targets = np.copy(targets)
        self.key = state_hash(self)

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
    new_state.map[state.actor[0]][state.actor[1]] = SPACE \
        if state.map[state.actor[0]][state.actor[1]] == ACTOR else TARGET

    # if next actor position is space
    if state.map[next_position[0]][next_position[1]] == SPACE:
        new_state.map[next_position[0]][next_position[1]] = ACTOR
    # if next actor position is box
    elif state.map[next_position[0]][next_position[1]] == BOX:
        # the position box will be pushed on
        next_two_position = next_position + actions[action]
        if state.map[next_two_position[0]][next_two_position[1]] in [SPACE, TARGET]:
            new_state.map[next_position[0]][next_position[1]] = ACTOR
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
        next_position = state.actor + actions[action]
        next_two_position = next_position + actions[action]
        if state.map[next_position[0], next_position[1]] in [SPACE, TARGET] or \
                (state.map[next_position[0], next_position[1]] == BOX and \
                    state.map[next_two_position[0], next_two_position[1]] not in [BOX, WALL, BOX_ON_TARGET]):
            feasible_actions.append(action)
    return feasible_actions

def get_reward(state, action, new_state):
    reward = 0
    next_position = state.actor + actions[action]

    if state.map[next_position[0], next_position[1]] in [SPACE, TARGET]:
        reward = BASIC_REWARD['SPACE']
    elif state.map[next_position[0], next_position[1]] in [BOX, BOX_ON_TARGET]:
        box_position = next_position + actions[action]
        box_next_position = box_position + actions[action]

        # push box off target
        if state.map[next_position[0], next_position[1]] == BOX_ON_TARGET:
            if state.map[box_position[0], box_position[1]] == SPACE:
                reward += BASIC_REWARD['OFF_TARGET']
            elif state.map[box_position[0], box_position[1]] == TARGET:
                reward += BASIC_REWARD['ON_TARGET']

        elif state.map[next_position[0], next_position[1]] == BOX:
            # infeasible push
            if state.map[box_position[0], box_position[1]] in [WALL, BOX, BOX_ON_TARGET]:
                reward += BASIC_REWARD['INFEASIBLE']
            elif state.map[box_position[0], box_position[1]] == TARGET:
                reward += BASIC_REWARD['ON_TARGET']
            elif state.map[box_position[0], box_position[1]] == SPACE:
                reward += BASIC_REWARD['ON_SPACE']
                # push box next to another box
                if state.map[box_next_position[0], box_next_position[1]] == BOX:
                    reward += BASIC_REWARD['BOX_BY_BOX']
                # push box next to wall
                elif state.map[box_next_position[0], box_next_position[1]] == WALL:
                    reward += BASIC_REWARD['BOX_BY_WALL']
        if is_goal(new_state):
            reward += BASIC_REWARD['GOAL']
        elif is_deadlock(new_state):
            reward += BASIC_REWARD['DEADLOCK']
    return reward

def is_goal(state):
    count = 0
    for t in state.targets:
        if state.map[t[0], t[1]] == TARGET:
            count += 1
    return count == 0

def is_deadlock(state):
    for loc in state.boxes:
        if is_immovable(state, loc) and state.map[loc[0], loc[1]] != BOX_ON_TARGET:
            return True
    return False

def is_out_of_bounds(state, loc):
    r, c = state.map.shape
    return (loc[0] < 0 or loc[0] >= r) and (loc[1] < 0 or loc[1] >= c)

# returns true if a box at loc is immovable
def is_immovable(state, loc):
    for a1, a2 in [('UP', 'RIGHT'), ('RIGHT', 'DOWN'), ('DOWN', 'LEFT'), ('LEFT', 'UP')]:
        n1 = loc + actions[a1]
        n2 = loc + actions[a2]

        # are n1, n2 occupied by wall/block?
        o1 = is_out_of_bounds(state, n1) or state.map[n1[0], n1[1]] in [WALL, BOX, BOX_ON_TARGET]
        o2 = is_out_of_bounds(state, n2) or state.map[n2[0], n2[1]] in [WALL, BOX, BOX_ON_TARGET]

        if o1 and o2:
            return True
    return False
