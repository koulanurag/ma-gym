import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..utils.draw import draw_grid, fill_cell, draw_cell_outline, draw_circle
import copy
from ..utils.action_space import MultiAgentActionSpace

logger = logging.getLogger(__name__)


class PongDuel(gym.Env):
    """"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        pass