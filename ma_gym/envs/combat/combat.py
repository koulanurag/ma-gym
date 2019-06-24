# -*- coding: utf-8 -*-

import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..utils.draw import draw_grid, fill_cell, draw_cell_outline, draw_circle
import copy
from ..utils.action_space import MultiAgentActionSpace

logger = logging.getLogger(__name__)


class Combat(gym.Env):
    """
    We simulate a simple battle involving two opposing teams in a 15×15 grid as shown in Fig. 2(middle).
    Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
    square around the team center, which is picked uniformly in the grid. At each time step, an agent can
    perform one of the following actions: move one cell in one of four directions; attack another agent
    by specifying its ID j (there are m attack actions, each corresponding to one enemy agent); or do
    nothing. If agent A attacks agent B, then B’s health point will be reduced by 1, but only if B is inside
    the firing range of A (its surrounding 3 × 3 area). Agents need one time step of cooling down after
    an attack, during which they cannot attack. All agents start with 3 health points, and die when their
    health reaches 0. A team will win if all agents in the other team die. The simulation ends when one
    team wins, or neither of teams win within 40 time steps (a draw).

    The model controls one team during training, and the other team consist of bots that follow a hardcoded policy.
    The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
    it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
    is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.
    When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
    encoding its unique ID, team ID, location, health points and cooldown. A model controlling an agent
    also sees other agents in its visual range (3 × 3 surrounding area). The model gets reward of -1 if the
    team loses or draws at the end of the game. In addition, it also get reward of −0.1 times the total
    health points of the enemy team, which encourages it to attack enemy bots.

    Reference : Learning Multiagent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(14, 14), n_agents=10, n_max=10, p_arrive=0.5, full_observable=False):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = 100
        self._step_count = None

        self.action_space = MultiAgentActionSpace([spaces.Discrete(2) for _ in range(self.n_agents)])
        self.agent_pos = {}

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._base_img = self.__draw_base_img()
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.viewer = None
        self._n_agents_routes = None
        self.full_observable = full_observable

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # agent id
            _agent_i_obs = [0 for _ in range(self.n_agents)]
            _agent_i_obs[agent_i] = 1

            # location
            _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def __draw_base_img(self):
        # create grid and make everything black
        img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=WALL_COLOR)

        # draw tracks
        for i, row in enumerate(self._base_grid):
            for j, col in enumerate(row):
                if col == PRE_IDS['empty']:
                    fill_cell(img, (i, j), cell_size=CELL_SIZE, fill='white', margin=0.05)
        return img

    def __create_grid(self):
        _grid = [[PRE_IDS['wall'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        _grid[self._grid_shape[0] // 2 - 1] = [PRE_IDS['empty'] for _ in range(self._grid_shape[1])]
        _grid[self._grid_shape[0] // 2] = [PRE_IDS['empty'] for _ in range(self._grid_shape[1])]

        for row in range(self._grid_shape[0]):
            _grid[row][self._grid_shape[1] // 2 - 1] = PRE_IDS['empty']
            _grid[row][self._grid_shape[1] // 2] = PRE_IDS['empty']

        return _grid

    def reset(self):
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._n_agents_routes = np.random.choice(range(len(self._routes)))
        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 30

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "GAS",
    1: "BRAKE",
}

PRE_IDS = {
    'wall': 'W',
    'empty': '0'
}
