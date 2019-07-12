import gym
import pytest
import ma_gym


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_CartPole-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 1


def test_reset(env):
    env.reset()
    assert env.env._step_count == 0
    assert env.env._total_episode_reward == 0
    assert env.env._agent_dones == [False for _ in range(env.n_agents)]
