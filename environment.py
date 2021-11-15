import gym
import numpy as np
import copy

SPACE = 0
ACTOR = 1
WALL = 2
BOX = 3
TARGET = 4
BOX_ON_TARGET = 5


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
    UP = np.array([-1, 0])
    DOWN = np.array([1, 0])
    LEFT = np.array([0, -1])
    RIGHT = np.array([0, 1])

    def __init__(self, config_text):
        self.state = State.from_config(config_text)
        self.history = []

    def step(self, action):
        """
        :param action:
        :return new state given this action:
        """
        new_state = copy.copy(self.state)
        next_position = self.state.actor + action

        # conditions when pushing:
        # 1. wall: can't move
        # 2. space: move
        # 3. box -> space: push
        # 4. box -> box/wall: can't move
        # 5. box -> target: move onto target

        if self.state.map[next_position[0]][next_position[1]] == SPACE:
            new_state.map[self.state.actor[0]][self.state.actor[1]] = SPACE
            new_state.map[next_position[0]][next_position[1]] = ACTOR
            new_state.actor = next_position

        elif self.state.map[next_position[0]][next_position[1]] == BOX:
            next_two_position = next_position + action
            if self.state.map[next_two_position[0]][next_two_position[1]] in [SPACE, TARGET]:
                new_state.map[self.state.actor[0]][self.state.actor[1]] = SPACE
                new_state.map[next_position[0]][next_position[1]] = ACTOR
                new_state.map[next_two_position[0]][next_two_position[1]] = BOX if \
                    self.state.map[next_two_position[0]][next_two_position[1]] == SPACE else BOX_ON_TARGET
                new_state.actor = next_position
                # modify box location been pushed
                for i in range(len(new_state.boxes)):
                    if np.array_equal(new_state.boxes[i], next_position):
                        new_state.boxes[i] = next_two_position
                        break
        # Save each state
        self.history.append(self.state)
        self.state = new_state

        return new_state

    def reset(self):
        self.state = self.history[0]
        self.history.clear()

    def render(self, mode="human"):
        if mode == "human":
            # TODO: using matplotlib
            pass
        return self.state.map

    def is_goal(self):
        pass

    def is_dead_end(self):
        # some boxes been pushed into corner
        pass
