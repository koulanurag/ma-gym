import argparse

import gym
import imageio
import numpy as np

import ma_gym


def parse_arguments():
    parser = argparse.ArgumentParser(description='Record the environment.')
    parser.add_argument('--output_dir', type=str, default='static/gif/',
                        help='Output directory with GIF record.')
    parser.add_argument('--env', type=str,
                        help='Name of recorded environment.')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames in GIF record.')
    parser.add_argument('--fps', type=int, default=21,
                        help='Frame per second.')
    return parser.parse_args()


def main(args):
    env = gym.make(args.env)
    pics = []
    done_n = [False] * env.n_agents

    obs_n = env.reset()
    while not all(done_n):
        pics.append(env.render(mode='rgb_array'))
        obs_n, _, done_n, _ = env.step(env.action_space.sample())

    print("Environment finished.")
    imageio.mimwrite(f'{args.env}.gif', pics[:args.frames], fps=args.fps)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
