import gym
import pytest
import ma_gym
from pytest_cases import parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_gym:PredatorPrey5x5-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_full():
    env = gym.make('ma_gym:PredatorPrey5x5-v1')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2
    assert env.n_preys == 1


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)]
    assert env._agent_dones == [False for _ in range(env.n_agents)]
    assert env._prey_alive == [True for _ in range(env.n_preys)]


@pytest.mark.parametrize('action_n,output', [([4, 4], [])])  # no-op action
def test_step(env, action_n, output):
    env.reset()
    obs_n, reward_n, done_n, info = env.step(action_n)
    assert env._step_count == 1
    assert env._total_episode_reward == reward_n, 'Total Episode reward doesn\'t match with one step reward'


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
