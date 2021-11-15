from environment import SokobanEnv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Put your file here")
    parser.add_argument("command", nargs="*")
    args = parser.parse_args()
    if len(args.command):
        with open(args.command[0], 'r') as f:
            content = "".join(f.readlines())

            env = SokobanEnv(config_text=content)
            print(env.state.map)
    else:
        print("No input file specified")

