import environment
import argparse
from qlearning import QLearner
import numpy as np

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
                ep_counts, sol_lens = [], []
                for i in range(10):
                    qlearner = QLearner(init_state)
                    ep_count, sol_len = qlearner.learn(episodes, display=False)
                    ep_counts.append(ep_count)
                    sol_lens.append(sol_len)
                    print(ep_count, sol_len)
                print(f"Episode count: mean={np.mean(ep_counts)}, min={np.min(ep_counts)}, max={np.max(ep_counts)}, std={np.std(ep_counts)}")
                print(f"Min episode length: mean={np.mean(sol_lens)}, min={np.min(sol_lens)}, max={np.max(sol_lens)}, std={np.std(sol_lens)}")
            else:
                qlearner = QLearner(init_state)
                qlearner.learn(episodes, display=True)
    else:
        print("No input file specified")

