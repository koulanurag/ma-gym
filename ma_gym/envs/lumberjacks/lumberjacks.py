import copy
import logging
import random
from dataclasses import dataclass
from typing import List, Mapping, Tuple, Union

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from PIL import Image, ImageColor, ImageDraw

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_circle, draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

Coordinates = Tuple[int, int]


@dataclass
class Agent:
    """Dataclass keeping all data for one agent/lumberjack in environment.

    Attributes:
        id: unique id in one environment run
        pos: position of the agent in grid
        reward: receive reward so far
        done: true whether the agent has finished
    """
    id: int
    pos: Coordinates
    reward: float
    done: bool


@dataclass
class Tree:
    """Dataclass keeping all data for one tree in environment.

    Attributes:
        id: unique id in one environment run
        pos: position of the tree in grid
        strength: number of hit-points (number of lumberjacks necessary to cut it down)
        alive: true if tree is not cut down
    """
    id: int
    pos: Coordinates
    strength: int
    alive: bool


EnvObject = Union[Agent, Tree]
Cell = List[EnvObject]


class Lumberjacks(gym.Env):
    """
    Lumberjacks environment involve a grid world, in which multiple lumberjacks attempt to cut down all trees. In order to cut down a tree in given cell, there must be present greater or equal number of agents/lumberjacks then the tree strength in the same location as tree. Tree is then cut down automatically.

    Agents select one of fire actions ∈ {No-Op, Down, Left, Up, Right}.
    Each agent's observation includes its:
        - agent ID (1)
        - position with in grid (2)
        - number of steps since beginning (1)
        - number of agents and tree strength for each cell in agent view (2 * `np.prod(agent_view_mask)`).
    All values are scaled down into range ∈ [0, 1].

    The terminating condition of this task is when all trees are cut down.

    Args:
        grid_shape: size of the grid
        n_agents: number of agents/lumberjacks
        n_trees: number of trees
        agent_view: size of the agent view range in each direction
        full_observable: flag whether agents should receive observation for all other agents
        step_cost: reward receive in each time step
        tree_cutdown_reward: reward received by agents who cut down the tree
        max_steps: maximum steps in one environment episode

    Attributes:
        _objects_dict: dictionary mapping from object ID (=unique identification within environment)
            and the environment object itself.
            Note that object ID is NOT arbitrary. IDs from 0 to `n_agents` are dedicated for agents IDs.
        _full_obs: dictionary mapping from grid position (represented with tuple of coordinates)
            to object IDs located on given position.
        _base_img: base image with grid
        _viewer: viewer for the rendered image
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape: Coordinates = (5, 5), n_agents: int = 2, n_trees: int = 12, agent_view: Tuple[int, int] = (1, 1), full_observable: bool = False, step_cost: float = -0.1, tree_cutdown_reward: float = 10, max_steps: int = 1000):
        assert 0 < n_agents
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_trees = n_trees
        self._agent_view = agent_view
        self._full_observable = full_observable
        self._step_cost = step_cost
        self._tree_cutdown_reward = tree_cutdown_reward
        self._max_steps = max_steps

        self._objects_dict = None  # Mapping[int, EnvObject]
        self._full_obs = None  # Mapping[Coordinates, List[int]]

        # Agent ID (1) + Pos (2) + Step (1) + Neighborhood (2 * mask_size)
        mask_size = np.prod(tuple(2 * v + 1 for v in agent_view))
        obs_high = np.array([1.] * (1 + 2 + 1 + 2 * mask_size))
        obs_low = np.array([0.] * (1 + 2 + 1 + 2 * mask_size))
        if self._full_observable:
            obs_high = np.tile(obs_high, self.n_agents)
            obs_low = np.tile(obs_low, self.n_agents)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(5)] * self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(obs_low, obs_high)] * self.n_agents)

        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        self._viewer = None

    def get_action_meanings(self, agent_id: int = None) -> Union[List[str], List[List[str]]]:
        """Returns list of actions meaning for `agent_id`.

        If `agent_id` is not specified returns meaning for all agents.
        """
        if agent_id is not None:
            assert agent_id <= self.n_agents
            return [k.upper() for k, v in sorted(ACTIONS_IDS.items(), key=lambda item: item[1])]
        else:
            return [[k.upper() for k, v in sorted(ACTIONS_IDS.items(), key=lambda item: item[1])]]

    def reset(self) -> List[List[float]]:
        self._objects_dict = {}
        self._init_full_obs()
        self._step_count = 0

        return self.get_agent_obs()

    def _init_full_obs(self):
        """Initialize environment for next episode.

        Fills `self._full_obs` and `self._objects_dict` with new values.
        """
        init_possitions = self._generate_init_pos()
        agent_id, tree_id = 0, self.n_agents
        self._full_obs = {index: [] for index in np.ndindex(*self._grid_shape)}

        for pos, cell in np.ndenumerate(init_possitions):
            if cell == PRE_IDS['agent']:
                self._objects_dict[agent_id] = Agent(id=agent_id, pos=pos, reward=0, done=False)
                self._get_cell(pos).append(agent_id)
                agent_id += 1
            elif cell == PRE_IDS['tree']:
                tree_id = tree_id
                self._objects_dict[tree_id] = Tree(
                    id=tree_id, pos=pos, strength=np.random.randint(1, self.n_agents + 1), alive=True)
                self._get_cell(pos).append(tree_id)
                tree_id += 1

    def _get_cell(self, pos) -> Cell:
        """Returns cell from grid in position `pos`. """
        return self._full_obs[tuple(pos)]

    def _generate_init_pos(self) -> np.ndarray:
        """Returns randomly selected initial positions for agents and trees.

        No agent or trees share the same cell in initial positions.
        """
        init_pos = np.array(
            [PRE_IDS['agent']] * self.n_agents +
            [PRE_IDS['tree']] * self._n_trees +
            [PRE_IDS['empty']] * (np.prod(self._grid_shape) - self.n_agents - self._n_trees)
        )
        np.random.shuffle(init_pos)
        return np.reshape(init_pos, self._grid_shape)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        for pos, cell in self._cell_generator():
            agent_strength, tree_strength = self._get_cell_strengths(cell)
            cell_size = (CELL_SIZE, CELL_SIZE / 2)

            if tree_strength != 0:
                tree_pos = (pos[0], 2 * pos[1])
                fill_cell(img, pos=tree_pos, cell_size=cell_size, fill=TREE_COLOR, margin=0.1)
                write_cell_text(img, text=str(tree_strength), pos=tree_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if agent_strength != 0:
                agent_pos = (pos[0], 2 * pos[1] + 1)
                draw_circle(img, pos=agent_pos, cell_size=cell_size, fill=AGENT_COLOR, radius=0.30)
                write_cell_text(img, text=str(agent_strength), pos=agent_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)
            return self._viewer.isopen

    def _get_cell_strengths(self, cell: Cell) -> Tuple[int, int]:
        """Returns tuple with number of agents and tree strength in given `cell`."""
        agent_strength, tree_strength = 0, 0
        for obj_id in cell:
            obj = self._objects_dict[obj_id]
            if isinstance(obj, Agent):
                agent_strength += 1
            elif isinstance(obj, Tree) and obj.alive:
                tree_strength += obj.strength

        return agent_strength, tree_strength

    def get_agent_obs(self) -> List[List[float]]:
        """Returns list of observations for each agent."""
        obs = []
        for agent_id, agent in self._agent_generator():
            agent_i_obs = [
                agent_id / self.n_agents,  # Agent ID
                agent.pos[0] / (self._grid_shape[0] - 1),  # Coordinate
                agent.pos[1] / (self._grid_shape[1] - 1),  # Coordinate
                self._step_count / self._max_steps,  # Steps
            ]

            for pos, cell in self._neighbour_generator(agent.pos, self._agent_view):
                agent_strength, tree_strength = self._get_cell_strengths(cell)
                agent_i_obs.append(agent_strength / self.n_agents)
                agent_i_obs.append(tree_strength / self.n_agents)
            obs.append(agent_i_obs)

        if self._full_observable:
            obs = [feature for agent_obs in obs for feature in agent_obs]
            obs = [obs] * self.n_agents

        return obs

    def _neighbour_generator(self, agent_pos: Coordinates, view_range: Tuple[int, int]) -> Tuple[Coordinates, Cell]:
        """Yields position and cell in neighborhood of the position `agent_pos` in range given by `view_range`.

        In yield value there is the cell of `agent_pos` as well. If the neighborhood position is out of the grid
        empty list is return.
        """
        for row in range(agent_pos[0] - view_range[0], agent_pos[0] + view_range[0] + 1):
            for col in range(agent_pos[1] - view_range[1], agent_pos[1] + view_range[1] + 1):
                pos = (row, col)
                if self._is_position_valid(pos):
                    yield pos, self._get_cell(pos)
                else:
                    yield pos, []

    def _is_position_valid(self, pos):
        return 0 <= pos[0] < self._grid_shape[0] and 0 <= pos[1] < self._grid_shape[1]

    def _agent_generator(self) -> Tuple[int, Agent]:
        """Yields agent_id and agent for all agents in environment."""
        for agent_id in range(self.n_agents):
            yield agent_id, self._objects_dict[agent_id]

    def _tree_generator(self) -> Tuple[int, Tree]:
        """Yields tree_id and tree for all trees in environment."""
        for agent_id in range(self.n_agents, self.n_agents + self._n_trees):
            yield agent_id, self._objects_dict[agent_id]

    def _cell_generator(self) -> Tuple[Coordinates, Cell]:
        """Yields position and cell for all cells in grid. """
        for pos in np.ndindex(*self._grid_shape):
            yield pos, self._get_cell(pos)

    def step(self, agents_action: List[int]):
        assert len(agents_action) == self.n_agents

        self._step_count += 1
        rewards = [self._step_cost] * self.n_agents

        # Move agents
        for (agent_id, agent), action in zip(self._agent_generator(), agents_action):
            if not agent.done:
                self._update_agent_pos(agent, action)

        # Cut down trees
        some_tree_alive = False
        for tree_id, tree in filter(lambda t: t[1].alive, self._tree_generator()):
            agent_ids = list(filter(
                lambda index: isinstance(self._objects_dict[index], Agent),
                self._get_cell(tree.pos)))
            if len(agent_ids) >= tree.strength:
                for agent_id in agent_ids:
                    rewards[agent_id] += self._tree_cutdown_reward
                tree.alive = False

            some_tree_alive |= tree.alive

        for agent_id, agent in self._agent_generator():
            agent.reward += rewards[agent_id]
            if (self._step_count >= self._max_steps) or (not some_tree_alive):
                agent.done = True

        return self.get_agent_obs(), rewards, [agent.done for _, agent in self._agent_generator()], {}

    def _update_agent_pos(self, agent: int, move: int):
        """Moves `agent` according the `move` command."""
        next_pos = self._next_pos(agent.pos, move)

        # Remove agent from old position
        self._get_cell(agent.pos).remove(agent.id)

        # Add agent to the new position
        agent.pos = next_pos
        self._get_cell(agent.pos).append(agent.id)

    def _next_pos(self, curr_pos: Coordinates, move: int) -> Coordinates:
        """Returns next valid position given by `move` command relative to `curr_pos`."""
        if move == ACTIONS_IDS['noop']:
            next_pos = curr_pos
        elif move == ACTIONS_IDS['down']:
            next_pos = (curr_pos[0] + 1, curr_pos[1])
        elif move == ACTIONS_IDS['left']:
            next_pos = (curr_pos[0], curr_pos[1] - 1)
        elif move == ACTIONS_IDS['up']:
            next_pos = (curr_pos[0] - 1, curr_pos[1])
        elif move == ACTIONS_IDS['right']:
            next_pos = (curr_pos[0], curr_pos[1] + 1)
        else:
            raise ValueError(f'Unknown action {move}.')
        return tuple(np.clip(next_pos, (0, 0), (self._grid_shape[0]-1, self._grid_shape[1]-1)))

    def seed(self, n: int):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
TREE_COLOR = 'green'
WALL_COLOR = 'black'

CELL_SIZE = 35

ACTIONS_IDS = {
    'noop': 0,
    'down': 1,
    'left': 2,
    'up': 3,
    'right': 4,
}

PRE_IDS = {
    'empty': 0,
    'wall': 1,
    'agent': 2,
    'tree': 3,
}
