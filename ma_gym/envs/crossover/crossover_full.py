import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..utils import draw_grid, fill_cell
import copy
from .crossover import CrossOver

logger = logging.getLogger(__name__)
import os, signal


class CrossOverF(CrossOver):
    """
    A simple environment to cross over the path of the other agent  (collaborative)

    Observation Space : Discrete
    Action Space : Discrete
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def step(self, *args, **kwargs):
        agent_partial_obs, rewards, dones, info = super().step(*args, **kwargs)
        full_obs = np.array(agent_partial_obs).flatten().tolist()
        full_obs = [full_obs for _ in range(self.n_agents)]
        return full_obs, rewards, dones, info

    def reset(self):
        agent_partial_obs = super().reset()
        full_obs = np.array(agent_partial_obs).flatten().tolist()
        full_obs = [full_obs for _ in range(self.n_agents)]
        return full_obs
