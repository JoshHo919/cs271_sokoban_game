from environment import SokobanEnv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Put your file here")
    parser.add_argument("command", nargs="*")
    args = parser.parse_args()
    if len(args.command):
        with open(args.command[0], 'r') as f:
            content = "".join(f.readlines())

            mode = 'auto'
            if input("Human?(Y/N) ") == 'Y':
                mode = 'human'
            env = SokobanEnv(config_text=content)

            episode = input("episode amount: ")

            for e in range(int(episode)):
                print("=========== episode " + str(e) + " ==================\n")
                if e != 0:
                    env.reset()
                while not env.is_deadlock() and not env.is_goal():
                    if mode == 'human':
                        env.step(input("input action: "))
                    else:
                        env.step(env.select_action())
                    # env.render()
                    print(env.state.map)

            for r in env.result:
                print(r)

            # for state_action, q_value in env.q_table.items():
            #     print(state_action[0].map, state_action[1])
            #     print(q_value)
    else:
        print("No input file specified")

