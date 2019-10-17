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
    assert env._step_count == 0
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)]
    assert env._agent_dones == [False for _ in range(env.n_agents)]


def test_reset_after_episode_end(env):
    env.reset()
    done = [False for _ in range(env.n_agents)]
    ep_reward = [0 for _ in range(env.n_agents)]
    step_i = 0
    while not all(done):
        step_i += 1
        _, reward_n, done, _ = env.step(env.action_space.sample())
        for i in range(env.n_agents):
            ep_reward[i] += reward_n[i]

    assert env._step_count == step_i
    assert env._total_episode_reward == ep_reward
    test_reset(env)


def test_observation_space(env):
    obs = env.reset()
    assert env.observation_space.contains(obs)
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        _, reward_n, done, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(env.observation_space.sample())
