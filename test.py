import gym
import ma_gym
import time

env = gym.make('PredatorPreyStable5x5-v0')

for _ in range(10):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        one_hot_actions = [[0 for _ in range(agent_action.n)] for agent_action in env.action_space]
        for agent_i, agent_action in enumerate(env.action_space):
            one_hot_actions[agent_i][agent_action.sample()] = 1
        print(obs,one_hot_actions)
        obs, _, done, _ = env.step(one_hot_actions)

        done = all(done)
        time.sleep(1)
