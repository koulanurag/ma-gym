import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ..utils import draw_grid, fill_cell, draw_cell_outline, draw_circle
import copy

logger = logging.getLogger(__name__)


class CrossOver(gym.Env):
    """
    A simple environment to cross over the path of the other agent  (collaborative)

    Observation Space : Discrete
    Action Space : Discrete
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.obs_size = 1
        self._grid_shape = (2, 8)
        self.n_agents = 2
        self._max_steps = 100
        self._step_count = None

        self.action_space = [spaces.Discrete(5) for _ in range(2)]  # l,r,t,d,noop
        self.final_agent_pos = {0: [0, 7], 1: [0, 0]}  # they have to go in opposite direction
        self.init_agent_pos = {0: [1, 2], 1: [1, 5]}

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self.__init_full_obs()
        self.__init_full_obs()
        self.viewer = None

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__wall_exists((row, col)):
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=WALL_COLOR)

        for agent_i, pos in self.final_agent_pos.items():
            row, col = pos[0], pos[1]
            draw_cell_outline(self._base_img, (row, col), cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])

    def __create_grid(self):
        _grid = np.zeros(self._grid_shape)
        _grid[1, :] = -1  # walls
        for agent_i, pos in self.init_agent_pos.items():
            _grid[pos[0], pos[1]] = 0
        return _grid

    def __init_full_obs(self):
        self.agent_pos = copy.copy(self.init_agent_pos)
        self._full_obs = self.__create_grid()
        for agent_i, pos in self.agent_pos.items():
            self.__update_agent_view(agent_i)
        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(0, self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]]
            _obs.append(_agent_i_obs)
        return _obs

    def reset(self):
        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return self._base_grid[row, col] == -1

    def _is_cell_vacant(self, pos):
        is_valid = (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])
        return is_valid and (self._full_obs[pos[0], pos[1]] == 0)

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0], curr_pos[1]] = 0
            self.__update_agent_view(agent_i)
        else:
            pass

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0], self.agent_pos[agent_i][1]] = agent_i + 1

    def __is_agent_done(self, agent_i):
        return self.agent_pos[agent_i] == self.final_agent_pos[agent_i]

    def step(self, one_hot_actions):
        self._step_count += 1
        rewards = [0 for _ in range(self.n_agents)]
        for agent_i, action in enumerate(one_hot_actions):
            action = action.index(1)
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

                self._agent_dones[agent_i] = self.__is_agent_done(agent_i)
                if self._agent_dones[agent_i]:
                    rewards[agent_i] = 5

        if self._step_count >= self._max_steps:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])
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


AGENT_COLORS = {
    0: 'red',
    1: 'blue'
}

CELL_SIZE = 15

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}
