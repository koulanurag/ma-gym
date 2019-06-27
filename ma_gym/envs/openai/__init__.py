import gym
from ..utils.action_space import MultiAgentActionSpace


class MultiAgentWrapper(gym.Wrapper):
    """ It's a multi agent wrapper over openai's single agent environments. """

    def __init__(self, name):
        super().__init__(gym.make(name))
        self.action_space = MultiAgentActionSpace([self.env.action_space])

    def step(self, action):
        action = action[0].index(1)
        obs, reward, done, info = self.env.step(action)
        return [obs], [reward], [done], [info]

    def reset(self):
        obs = self.env.reset()
        return [obs]

    def close(self):
        return self.env.close()
