import gym
import numpy as np
import pytest
from pytest_cases import pytest_parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_gym:TrafficJunction-v0')
    yield env
    env.close()


def test_init(env):
    assert 1 <= env._n_max <= 10, 'N_max must be between 1 and 10, got {}'.format(env._n_max)
    assert env._on_the_road.count(True) <= len(env._entry_gates), 'Cars on the road after initializing cannot ' \
                                                                  'be higher than {}'.format(len(env._entry_gates))


def test_reset(env):
    env.reset()
    assert env._step_count == 0, 'Step count should be 0 after reset, got {}'.format(env._step_count)
    assert env._agent_step_count == [0 for _ in range(env.n_agents)], 'Agent step count should be 0 for all agents' \
                                                                      ' after reset'
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)], 'Total reward should be 0 after reset'
    assert env._agent_dones == [False for _ in range(env.n_agents)], 'Agents cannot be done when the environment' \
                                                                     ' resets'
    assert env._agent_turned == [False for _ in range(env.n_agents)], 'Agents cannot have changed direction ' \
                                                                      ' when the environment resets'


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

    assert step_i == env._step_count, 'Number of steps after an episode must match ' \
                                      'env._step_count, got {} and {}'.format(step_i, env._step_count)
    assert env._total_episode_reward == ep_reward, 'Total reward after an episode must match' \
                                                   ' env._total_episode_reward, ' \
                                                   'got {} and {}'.format(ep_reward, env._total_episode_reward)
    test_reset(env)


@pytest_parametrize_plus('env',
                         [fixture_ref(env)])
def test_observation_space(env):
    obs = env.reset()
    expected_agent_i_shape = (np.prod(env._agent_view_mask) * (env.n_agents + 2 + 3),)
    for agent_i in range(env.n_agents):
        assert obs[agent_i].shape == expected_agent_i_shape, \
            'shape of obs. expected to be {}; but found to be {}'.format(expected_agent_i_shape, obs[agent_i].shape)

    assert env.observation_space.contains(obs), 'Observation must be part of the observation space'
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        obs, reward_n, done, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), 'Observation must be part of the observation space'
    assert env.observation_space.contains(obs), 'Observation must be part of the observation space'
    assert env.observation_space.contains(env.observation_space.sample()), 'Observation must be part of the' \
                                                                           ' observation space'


@pytest_parametrize_plus('env',
                         [fixture_ref(env)])
def test_step_cost(env):
    env.reset()
    for step_i in range(3):
        obs, reward_n, done, _ = env.step(env.action_space.sample())
        target_reward = [env._step_cost * (step_i + 1) for _ in range(env.n_agents)]
        assert (reward_n == target_reward), \
            'step_cost is not correct. Expected {} ; Got {}'.format(target_reward, reward_n)
