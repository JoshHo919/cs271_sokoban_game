import gym
import numpy as np
import copy
import matplotlib.pyplot as plt
import random

SPACE = 0
ACTOR = 1
WALL = 2
BOX = 3
TARGET = 4
BOX_ON_TARGET = 5
ACTOR_ON_TARGET = 6

BASIC_REWARD = {'SPACE': -1, 'BOX_BY_BOX': -40, 'BOX_BY_WALL': -20, 'INFEASIBLE': -500,\
                'ON_TARGET': 300, 'ON_SPACE': -1, 'OFF_TARGET': -400, 'DEADLOCK': -500}

UP = np.array([-1, 0])
DOWN = np.array([1, 0])
LEFT = np.array([0, -1])
RIGHT = np.array([0, 1])
actions = {'UP': UP, 'LEFT': LEFT, 'DOWN': DOWN, 'RIGHT': RIGHT}

epsilon = 0.1
maximum_length = 1000


class State:
    def __init__(self, map_array, actor, boxes, targets):
        self.map = np.copy(map_array)
        self.actor = np.copy(actor)
        self.boxes = np.copy(boxes)
        self.targets = np.copy(targets)

    @classmethod
    def from_config(cls, config_text="""5 3
12 1 1 1 2 1 3 2 1 2 3 3 1 3 3 4 1 4 3 5 1 5 2 5 3
1 3 2
1 4 2
2 2"""):
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
        for i in range(1, int(box_str[0]) + 1, 2):
            np_map[int(box_str[i]) - 1, int(box_str[i + 1]) - 1] = BOX
            boxes.append([int(box_str[i]) - 1, int(box_str[i + 1]) - 1])
        np_boxes = np.array(boxes)

        targets = []
        for i in range(1, int(target_str[0]) + 1, 2):
            if np_map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] == BOX:
                np_map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] = BOX_ON_TARGET
            else:
                np_map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] = TARGET
            targets.append([int(target_str[i]) - 1, int(target_str[i + 1]) - 1])
        np_targets = np.array(targets)

        np_map[int(actor_str[0]) - 1, int(actor_str[1]) - 1] = ACTOR

        return cls(np_map, np_actor, np_boxes, np_targets)

    def __copy__(self):
        return State(self.map, self.actor, self.boxes, self.targets)


class SokobanEnv(gym.Env):

    def __init__(self, config_text):
        self.state = State.from_config(config_text)
        self.history = []
        self.q_table = {}
        self.f_table = {}

        self.current_length = 0
        self.discount_factor = 0.5
        self.deadlock = False

        self.episode = 0
        self.result = []

    def select_action(self):
        feasible_actions = []

        for name, action in actions.items():
            next_position = self.state.actor + action
            next_two_position = next_position + action

            # wall and unmovable box are unfeasible actions
            if self.state.map[next_position[0], next_position[1]] in [SPACE, TARGET] or \
                    (self.state.map[next_position[0], next_position[1]] == BOX and \
                     self.state.map[next_two_position[0], next_two_position[1]] not in [BOX, WALL, BOX_ON_TARGET]):
                q_value = 0
                if (self.state, name) in self.q_table:
                    q_value = self.q_table[(self.state, name)]
                feasible_actions.append((name, q_value))

        # apply exploration with epsilon-greedy
        if random.random() <= epsilon:
            final_action = random.choice(feasible_actions)[0]
        else:
            max_q = max(feasible_actions, key=lambda item: item[1])[1]
            final_action = random.choice([action[0] for action in feasible_actions if action[1] == max_q])

        print("Selected action: " + final_action)
        return final_action

    def step(self, action):
        """
        :param action: (str) action
        """

        new_state = copy.copy(self.state)
        next_position = self.state.actor + actions[action]

        # Update F-table
        if (self.state, action) in self.f_table:
            self.f_table.update({(self.state, action): self.f_table[(self.state, action)] + 1})
        else:
            self.f_table.update({(self.state, action): 1})

        # conditions when pushing:
        # 1. wall: can't move
        # 2. space: move
        # 3. box -> space: push
        # 4. box -> box/wall: can't move
        # 5. box -> target: move onto target

        # update actor's current position for the new state depends on current actor on target or space
        new_state.map[self.state.actor[0]][self.state.actor[1]] = SPACE \
            if self.state.map[self.state.actor[0]][self.state.actor[1]] == ACTOR else TARGET

        # if next actor position is space
        if self.state.map[next_position[0]][next_position[1]] == SPACE:
            new_state.map[next_position[0]][next_position[1]] = ACTOR
        # if next actor position is box
        elif self.state.map[next_position[0]][next_position[1]] == BOX:
            # the position box will be pushed on
            next_two_position = next_position + actions[action]
            if self.state.map[next_two_position[0]][next_two_position[1]] in [SPACE, TARGET]:
                new_state.map[next_position[0]][next_position[1]] = ACTOR
                new_state.map[next_two_position[0]][next_two_position[1]] = BOX if \
                    self.state.map[next_two_position[0]][next_two_position[1]] == SPACE else BOX_ON_TARGET
                # modify box location been pushed
                for i in range(len(new_state.boxes)):
                    if np.array_equal(new_state.boxes[i], next_position):
                        new_state.boxes[i] = next_two_position
                        break
            # check if the pushed box is blocked
            self.deadlock = True if self.is_blocked(new_state, next_two_position, action) else False
            # print(self.deadlock)
        elif self.state.map[next_position[0]][next_position[1]] == TARGET:
            new_state.map[next_position[0]][next_position[1]] = ACTOR_ON_TARGET

        # update actor position in the new state
        new_state.actor = next_position

        # get current reward
        reward = self.get_reward(self.state, action)
        # get maximum future reward
        future_reward = self.observe(new_state)
        # get learning rate
        learning_rate = 1 / (2 * (self.f_table[(self.state, action)] + 1))

        # calculate Q-value
        if (self.state, action) in self.q_table:
            new_q_value = self.q_table[(self.state, action)] + \
                          learning_rate * (reward + self.discount_factor * future_reward - self.q_table[
                (self.state, action)])
        else:
            new_q_value = learning_rate * (reward + self.discount_factor * future_reward)

        # update Q-value
        self.q_table.update({(self.state, action): new_q_value})

        print("Q_value: " + str(new_q_value))

        # Save each state, current state <- new state
        self.history.append(self.state)
        self.state = new_state
        self.current_length += 1

    def observe(self, state):
        """
        :param state: a future state
        :return: the maximum reward of the future state
        """
        rewards = []
        for action in actions:
            rewards.append(self.get_reward(state, action))

        print("future reward: " + str(max(rewards)))
        return max(rewards)

    def get_reward(self, state, action):
        """

        :param state: a given state
        :param action: a given action
        :return: the reward found with given (SxA)
        """
        reward = 0
        next_position = state.actor + actions[action]

        if state.map[next_position[0], next_position[1]] in [SPACE, TARGET]:
            reward = BASIC_REWARD['SPACE']
        elif state.map[next_position[0], next_position[1]] in [BOX, BOX_ON_TARGET]:
            box_position = next_position + actions[action]
            box_next_position = box_position + actions[action]

            # print("actor position: ", state.actor)
            # print("box will be pushed to: ", box_position)
            # print("position behind box: ", box_next_position)

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
        # if deadlock triggered
        if self.deadlock:
            reward += BASIC_REWARD['DEADLOCK']
        return reward

    def reset(self):
        self.state = self.history[0]
        self.history.clear()

        self.current_length = 0
        self.deadlock = False
        self.episode += 1

    def render(self, mode="human"):
        if mode == "human":
            # Use matplotlib to plot the map
            plt.imshow(self.state.map, interpolation='none')
            plt.show()
            pass
        return self.state.map

    def is_deadlock(self):
        if self.current_length >= 1000:
            self.deadlock = True
        if self.deadlock:
            print("--- DEADLOCK ---")
            self.result.append((self.episode, 'deadlock'))
            return True

    def is_goal(self):
        count = 0
        for row in self.state.map:
            for col in row:
                if col == 4:
                    count += 1
        if count == 0:
            print("--- GOAL ---")
            self.result.append((self.episode, 'goal'))
            return True
        else:
            return False

    def is_blocked(self, state, box_position, action):
        """
        Check if the box is in a deadlock position
        :param state: A given state
        :param box_position: the position of a specific box
        :return: A boolean value whether the box is blocked(deadlock)
        """
        # check if the box has been pushed into corner
        blocked_X = 0
        blocked_Y = 0

        # check if the box ON TARGET
        if state.map[box_position[0], box_position[1]] == BOX_ON_TARGET:
            return False
        for direction in actions:
            neighbor = box_position + actions[direction]
            if state.map[neighbor[0], neighbor[1]] in [2, 3]:
                if direction in ['UP', 'DOWN']:
                    blocked_Y += 1
                else:
                    blocked_X += 1

        box_next_position = box_position + actions[action]
        # if box by box will cause deadlock
        if state.map[box_next_position[0], box_next_position[1]] == BOX:
            next_blocked_X = 0
            next_blocked_Y = 0
            for direction in actions:
                neighbor = box_position + actions[direction]
                if state.map[neighbor[0], neighbor[1]] in [2, 3]:
                    if direction in ['UP', 'DOWN']:
                        next_blocked_Y += 1
                    else:
                        next_blocked_X += 1
            if action in ['UP', 'DOWN'] and blocked_X >= 1 and next_blocked_X >= 1:
                return True
            elif action in ['LEFT', 'RIGHT'] and blocked_Y >= 1 and next_blocked_Y >= 1:
                return True

        elif blocked_X >= 1 and blocked_Y >= 1:
            return True
        return False
