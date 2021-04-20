# -*- coding: utf-8 -*-

import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell

logger = logging.getLogger(__name__)


class TrafficJunction(gym.Env):
    """
    This consists of a 4-way junction on a 14 × 14 grid. At each time step, new cars enter the grid with
    probability `p_arrive` from each of the four directions. However, the total number of cars at any given
    time is limited to `Nmax = 10`.

    Each car occupies a single cell at any given time
    and is randomly assigned to one of three possible routes (keeping to the right-hand side of the road).
    At every time step, a car has two possible actions: gas which advances it by one cell on its route or
    brake to stay at its current location. A car will be removed once it reaches its destination at the edge
    of the grid.

    Two cars collide if their locations overlap. A collision incurs a reward `rcoll = −10`, but does not affect
    the simulation in any other way. To discourage a traffic jam, each car gets reward of `τrtime = −0.01τ`
    at every time step, where `τ` is the number time steps passed since the car arrived. Therefore, the total
    reward at time t is

    Each car is represented by one-hot binary vector set {n, l, r}, that encodes its unique ID, current location
    and assigned route number respectively. Each agent controlling a car can only observe other cars in its vision
    range (a surrounding 3 × 3 neighborhood), but it can communicate to all other cars.

    The state vector s_j for each agent is thus a concatenation of all these vectors, having dimension
    32 × |n| × |l| × |r|.

    Reference : Learning Multiagent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(14, 14), n_agents=10, n_max=10, arrive_prob=0.5, full_observable=False):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = 100
        self._step_count = None
        self._collision_reward = -10
        self._total_episode_reward = None
        self._arrive_prob = arrive_prob
        self._n_max = n_max

        self._entry_gates = [(7, 0), (13, 7), (0, 6), (6, 13)]

        self.action_space = MultiAgentActionSpace([spaces.Discrete(2) for _ in range(self.n_agents)])
        self.agent_pos = {}

        self._full_obs = self.__create_grid()
        self._base_img = self.__draw_base_img()
        self._agent_dones = [None for _ in range(self.n_agents)]

        self.viewer = None
        self._n_agents_routes = None
        self.full_observable = full_observable

        # agent id (n_agents, onehot), pos (2)
        self._obs_high = np.array([1.0] * self.n_agents + [1.0, 1.0], dtype=np.float32)
        self._obs_low = np.array([0.0] * self.n_agents + [0.0, 0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])
        self.seed()

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

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
        for i, row in enumerate(self._full_obs):
            for j, col in enumerate(row):
                if col == PRE_IDS['empty']:
                    fill_cell(img, (i, j), cell_size=CELL_SIZE, fill='white', margin=0.05)
        return img

    def __create_grid(self):
        # create a grid with every cell as wall
        _grid = [[PRE_IDS['wall'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]

        # draw track by making cells empty :
        # horizontal tracks
        _grid[self._grid_shape[0] // 2 - 1] = [PRE_IDS['empty'] for _ in range(self._grid_shape[1])]
        _grid[self._grid_shape[0] // 2] = [PRE_IDS['empty'] for _ in range(self._grid_shape[1])]

        # vertical tracks
        for row in range(self._grid_shape[0]):
            _grid[row][self._grid_shape[1] // 2 - 1] = PRE_IDS['empty']
            _grid[row][self._grid_shape[1] // 2] = PRE_IDS['empty']

        return _grid

    def reset(self):
        self._total_episode_reward = 0
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        self._full_obs = self.__create_grid()
        self.curr_cars_count = 0

        # sample cars for each location
        for gates in self._entry_gates:
            if self.curr_cars_count <= self._n_max and self.np_random.random() < self._arrive_prob:
                pass

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

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

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
