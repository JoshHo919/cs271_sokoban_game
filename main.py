import environment
import argparse
from qlearning import QLearner
from time import perf_counter
import numpy as np

def print_data(name, data):
    print(f"{name}: mean={np.mean(data)}, min={np.min(data)}, max={np.max(data)}, std={np.std(data)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Put your file here")
    parser.add_argument("command", nargs="*")
    args = parser.parse_args()
    if len(args.command):
        with open(args.command[0], 'r') as f:
            config_text = "".join(f.readlines())
            episodes = 1000
            init_state = environment.State.from_config(config_text)

            testing = False

            if testing:
                ep_counts, sol_lens, times = [], [], []
                for i in range(10):
                    t0 = perf_counter()
                    qlearner = QLearner(init_state)
                    ep_count, sol_len = qlearner.learn(episodes, display=False)
                    t1 = perf_counter()
                    ep_counts.append(ep_count)
                    sol_lens.append(sol_len)
                    times.append((t1 - t0) * 1000)
                    print(ep_count, sol_len, (t1-t0) * 1000)
                print_data("Episode count", ep_counts)
                print_data("Min episode length", sol_lens)
                print_data("Time", times)
            else:
                qlearner = QLearner(init_state)
                n, actions = qlearner.learn(episodes, display=False)
                print(f"{n} {' '.join(map(lambda x: x[0], actions))}")
    else:
        print("No input file specified")

