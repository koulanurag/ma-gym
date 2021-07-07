import gym
import pytest
from pytest_cases import parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_gym:PredatorPrey7x7-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_full():
    env = gym.make('ma_gym:PredatorPrey7x7-v1')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 4


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)]
    assert env._agent_dones == [False for _ in range(env.n_agents)]
    assert env._prey_alive == [True for _ in range(env.n_preys)]


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
    assert env._total_episode_reward == ep_reward
    test_reset(env)


@parametrize_plus('env', [fixture_ref(env),
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
