import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..utils.draw import draw_grid, fill_cell, draw_cell_outline, draw_circle
import copy
from ..utils.action_space import MultiAgentActionSpace
import random

logger = logging.getLogger(__name__)


class PongDuel(gym.Env):
    """Two Player Pong Game"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self._grid_shape = (50, 40)
        self.n_agents = 2
        self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(self.n_agents)])

        self._step_count = None
        self.__total_episode_reward = None
        self.agent_pos = None
        self.ball_pos = None

        self.curr_ball_dir = None

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_ball_view(self):
        self._full_obs[self.ball_pos[0], self.ball_pos[1]] = PRE_IDS['ball']

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()
        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)

        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)

        self.__update_ball_view(agent_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        pass

    def reset(self):
        self.agent_pos[0] = (random.randint(2, self._grid_shape[0] - 3), 1)
        self.agent_pos[1] = (random.randint(2, self._grid_shape[0] - 3), self._grid_shape[1] - 2)

        self.ball_pos = (random.randint(5, self._grid_shape[0] - 5), random.randint(10, self._grid_shape[1] - 10))
        self.curr_ball_dir = random.choice(BALL_DIRECTIONS)

        self.__init_full_obs()
        self._step_count = 0
        self.__total_episode_reward = 0

        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for y in range(self.agent_pos[agent_i] - 2, self.agent_pos + 3):
                fill_cell(img, (self.agent_pos[agent_i][0], y), cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])

        fill_cell(img, self.ball_pos[0], cell_size=CELL_SIZE, fill=BALL_HEAD_COLOR)
        fill_cell(img, self.ball_pos[1], cell_size=CELL_SIZE, fill=BALL_HEAD_COLOR)
        fill_cell(img, self.ball_pos[2], cell_size=CELL_SIZE, fill=BALL_HEAD_COLOR)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def step(self, action_n):
        pass


CELL_SIZE = 10

ACTION_MEANING = {
    0: "NOOP",
    1: "UP",
    2: "DOWN",
}

AGENT_COLORS = {
    0: 'red',
    1: 'blue'
}
WALL_COLOR = 'black'
BALL_HEAD_COLOR = 'orange'
BALL_TAIL_COLOR = 'yellow'

# each pre-id should be unique and single char
PRE_IDS = {
    'agent': 'A',
    'wall': 'W',
    'ball': 'B',
}

BALL_DIRECTIONS = ['NW', 'W', 'SW', 'SE', 'E', 'NE']
