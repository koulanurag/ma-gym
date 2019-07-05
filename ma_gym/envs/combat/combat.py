# -*- coding: utf-8 -*-

import copy
import logging
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, write_cell_text

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

    def __init__(self, grid_shape=(15, 15), n_agents=5, n_opponents=5, init_health=3, full_observable=False):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_opponents = n_opponents
        self._max_steps = 40
        self._step_cost = 0
        self._step_count = None

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(5 + self._n_opponents) for _ in range(self.n_agents)])

        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_prev_pos = {_: None for _ in range(self.n_agents)}
        self.opp_pos = {_: None for _ in range(self.n_agents)}
        self.opp_prev_pos = {_: None for _ in range(self.n_agents)}

        self._init_health = init_health
        self.agent_health = {_: None for _ in range(self.n_agents)}
        self.opp_health = {_: None for _ in range(self._n_opponents)}
        self._agent_dones = [None for _ in range(self.n_agents)]

        self.viewer = None
        self.full_observable = full_observable

    def get_agent_obs(self):
        pass

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_opp_view(self, opp_i):
        self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        # randomly select agent pos
        for agent_i in range(self.n_agents):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.agent_prev_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    self.__update_agent_view(agent_i)
                    break

        # randomly select opponent pos
        for opp_i in range(self._n_opponents):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.opp_prev_pos[opp_i] = pos
                    self.opp_pos[opp_i] = pos
                    self.__update_opp_view(opp_i)
                    break

        self.__draw_base_img()

    def reset(self):
        self._step_count = 0
        self.__total_episode_reward = 0
        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.__init_full_obs()
        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        for agent_i in range(self.n_agents):
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.3)
        for opp_i in range(self._n_opponents):
            fill_cell(img, self.opp_pos[opp_i], cell_size=CELL_SIZE, fill=OPPONENT_COLOR)
            write_cell_text(img, text=str(opp_i + 1), pos=self.opp_pos[opp_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.3)

        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

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
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_opp_pos(self, opp_i, move):

        curr_pos = copy.copy(self.opp_pos[opp_i])
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
            self.opp_pos[opp_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_opp_view(opp_i)

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def step(self, agents_action):
        assert len(agents_action) == self.n_agents

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i]>0:
                if action <= 4:
                    self.__update_agent_pos(agent_i, action)
                else:
                    pass

        for opp_i, action in enumerate(agents_action):
            if self.opp_health[opp_i]>0:
                if action <= 4:
                    self.__update_opp_pos(opp_i, action)
                else:
                    pass

        # step overflow or all opponents dead
        if self._step_count >= self._max_steps or sum(self.opp_health) == 0:
            self._agent_dones = [True for _ in range(self.n_agents)]

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 15

WALL_COLOR = 'black'
AGENT_COLOR = 'red'
OPPONENT_COLOR = 'blue'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'wall': 'W',
    'empty': '0',
    'agent': 'A',
    'opponent': 'X',
}
