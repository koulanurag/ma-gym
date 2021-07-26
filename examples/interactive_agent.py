import argparse

import gym

from ma_gym.wrappers import Monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Agent for ma-gym')
    parser.add_argument('--env', default='Checkers-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    print('Enter the actions space together and press enter ( Eg: \'11<enter>\' which meanes take 1'
          ' for agent 1 and 1 for agent 2)')

    env = gym.make('ma_gym:{}'.format(args.env))
    env = Monitor(env, directory='recordings', force=True)
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        obs_n = env.reset()
        env.render()
        while not all(done_n):
            action_n = [int(_) for _ in input('Action:')]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            ep_reward += sum(reward_n)
            env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()
