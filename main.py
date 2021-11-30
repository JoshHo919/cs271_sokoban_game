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

            episode = int(input("episode amount: "))

            for e in range(episode):
                print("=========== episode " + str(e) + " ==================\n")
                if e != 0:
                    env.reset()
                while not env.is_deadlock() and not env.is_goal():
                    # print(env.state.map)
                    if mode == 'human':
                        env.step(input("input action: "))
                    else:
                        env.step(env.select_action())
                    # env.render()


            goal_rate = 0
            last100_goal_rate = 0
            for r in env.result:
                if r[1] == 'goal':
                    goal_rate += 1
                    if r[0] >= episode - 100:
                        last100_goal_rate += 1
                print(r)
            print("Goal rate: " + str((goal_rate/episode)*100) + '%')
            print("Last 100 goal rate: " + str((last100_goal_rate / 100) * 100) + '%')
            print("Q-FIND: ", env.q_find)
            # for state_action, q_value in env.q_table.items():
            #     print(state_action[0].map, state_action[1])
            #     print(q_value)
    else:
        print("No input file specified")

