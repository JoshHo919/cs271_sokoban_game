import environment
import argparse
from qlearning import QLearner

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Put your file here")
    parser.add_argument("command", nargs="*")
    args = parser.parse_args()
    if len(args.command):
        with open(args.command[0], 'r') as f:
            config_text = "".join(f.readlines())
            episodes = 1000
            init_state = environment.State.from_config(config_text)
            qlearner = QLearner(init_state)
            qlearner.learn(episodes, display=True)
    else:
        print("No input file specified")

