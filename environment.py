import gym
import numpy as np

SPACE = 0
ACTOR = 1
WALL = 2
BOX = 3
TARGET = 4
BOX_ON_TARGET = 5


class State:
    def __init__(self, config_text="""5 3
12 1 1 1 2 1 3 2 1 2 3 3 1 3 3 4 1 4 3 5 1 5 2 5 3
1 3 2
1 4 2
2 2"""):
        self.config_text = config_text
        lines = config_text.split("\n")
        dimensions = lines[0].split(" ")
        wall_str = lines[1].split(" ")
        box_str = lines[2].split(" ")
        target_str = lines[3].split(" ")
        actor_str = lines[4].split(" ")

        self.map = np.zeros((int(dimensions[0]), int(dimensions[1])))
        self.actor = np.array([int(actor_str[0]), int(actor_str[1])])
        self.boxes = np.array([])
        self.targets = np.array([])

        for i in range(1, int(wall_str[0]) + 1):
            print(int(wall_str[i * 2 - 1]) - 1, int(wall_str[i * 2]) - 1)
            self.map[int(wall_str[i * 2 - 1]) - 1, int(wall_str[i * 2]) - 1] = WALL

        boxes = []
        for i in range(1, int(box_str[0]) + 1, 2):
            self.map[int(box_str[i]) - 1, int(box_str[i + 1]) - 1] = BOX
            boxes.append([int(box_str[i]) - 1, int(box_str[i + 1]) - 1])
        self.boxes = np.array(boxes)

        targets = []
        for i in range(1, int(target_str[0]) + 1, 2):
            if self.map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] == BOX:
                self.map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] = BOX_ON_TARGET
            else:
                self.map[int(target_str[i]) - 1, int(target_str[i + 1]) - 1] = TARGET
            targets.append([int(target_str[i]) - 1, int(target_str[i + 1]) - 1])
        self.targets = np.array(targets)

        self.map[int(actor_str[0]) - 1, int(actor_str[1]) - 1] = ACTOR

    def __copy__(self):
        return State(self.config_text)



class SokobanEnv(gym.Env):
    def __init__(self):
        # self.state =
        pass

    def step(self, action):
        if action == 1:
            self.state = 1
        else:
            self.state = 0

    def reset(self):
        pass

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    s = State()
    print(s.map)
    print(s.boxes)
    print(s.targets)
    print(s.actor)
