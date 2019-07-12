import gym
import pytest
import ma_gym


@pytest.fixture(scope='module')
def env():
    env = gym.make('PredatorPrey5x5-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2
    assert env.n_preys == 1


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert env._total_episode_reward == 0
    assert env._agent_dones == [False for _ in range(env.n_agents)]
    assert env._prey_alive == [True for _ in range(env.n_preys)]


@pytest.mark.parametrize('action_n,output', [([4, 4], [])])  # no-op action
def test_step(env, action_n, output):
    env.reset()
    obs_n, reward_n, done_n, info = env.step(action_n)
    assert env._step_count == 1
    assert env._total_episode_reward == sum(reward_n), 'Total Episode reward doesn\'t match with one step reward'

