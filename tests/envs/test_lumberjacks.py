import gym
import numpy as np
import pytest
from pytest_cases import fixture_ref, parametrize_plus

import ma_gym


@pytest.fixture(scope='module')
def env():
    env = gym.make('Lumberjacks-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_full():
    env = gym.make('Lumberjacks-v1')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2
    assert env._n_trees == 12
    assert env._agent_view == (1, 1)


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert len(env._agents) == np.sum(env._agent_map)== 2
    assert np.sum(env._tree_map > 0) == 12
    for agent_id, agent in env._agent_generator():
        assert env._agent_dones[agent_id] == False, 'Game cannot finished after reset'
        assert env._total_episode_reward[agent_id] == 0 , 'Total Episode reward doesn\'t match with one step reward'
    assert np.sum((env._tree_map < 0) & (env._tree_map > 2)) == 0


@pytest.mark.parametrize('action_n', [[0, 0]])  # no-op action
def test_step(env, action_n):
    env.reset()
    obs_n, reward_n, done_n, info = env.step(action_n)

    assert env._step_count == 1
    for (agent_id, agent), reward in zip(env._agent_generator(), reward_n):
        assert env._agent_dones[agent_id] == False, 'Game cannot finished after one step'
        assert env._total_episode_reward[agent_id] == reward, 'Total Episode reward doesn\'t match with one step reward'


def test_reset_after_episode_end(env):
    env.reset()
    done = [False for _ in range(env.n_agents)]
    step_i = 0
    ep_reward = [0 for _ in range(env.n_agents)]
    while not all(done):
        step_i += 1
        _, reward_n, done, _ = env.step(env.action_space.sample())
        for i in range(env.n_agents):
            ep_reward[i] += reward_n[i]

    assert step_i == env._step_count
    for (agent_id, agent), reward in zip(env._agent_generator(), ep_reward):
        assert env._agent_dones[agent_id] == True
        assert env._total_episode_reward[agent_id] == reward, 'Total Episode reward doesn\'t match with one step reward'
    test_reset(env)


@parametrize_plus('env',
                  [fixture_ref(env),
                   fixture_ref(env_full)])
def test_observation_space(env):
    obs = env.reset()
    assert env.observation_space.contains(obs)
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        obs, reward_n, done, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(env.observation_space.sample())
