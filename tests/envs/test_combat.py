import gym
import pytest
import ma_gym


@pytest.fixture(scope='module')
def env():
    env = gym.make('Combat-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 5


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert env._total_episode_reward == 0
    assert env._agent_dones == [False for _ in range(env.n_agents)]


def test_reset_after_episode_end(env):
    env.reset()
    done = [False for _ in range(env.n_agents)]
    step_i = 0
    while not all(done):
        step_i += 1
        _, _, done, _ = env.step(env.action_space.sample())
    assert step_i == env._step_count
    test_reset(env)
