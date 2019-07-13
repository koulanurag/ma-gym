import gym
import pytest
import ma_gym


@pytest.fixture(scope='module')
def env():
    env = gym.make('Switch2-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2


def test_reset(env):
    obs_n = env.reset()

    target_obs_n = [[0, 0.17],
                    [0, 0.83]]
    assert env._step_count == 0
    assert env._total_episode_reward == 0
    assert env._agent_dones == [False for _ in range(env.n_agents)]

    for i in range(env.n_agents):
        assert obs_n[i] == target_obs_n[i]


def test_reset_after_episode_end(env):
    env.reset()
    done = [False for _ in range(env.n_agents)]
    step_i = 0
    while not all(done):
        step_i += 1
        _, _, done, info = env.step(env.action_space.sample())

    assert step_i == env._step_count
    test_reset(env)


@pytest.mark.parametrize('action_n,output',
                         [([1, 1],  # action
                           ([[0.0, 0.00], [0, 0.83]])  # obs
                           )])
def test_step(env, action_n, output):
    env.reset()
    target_obs_n = output
    obs_n, reward_n, done_n, info = env.step(action_n)

    assert env._step_count == 1
    assert env._total_episode_reward == sum(reward_n), 'Total Episode reward doesn\'t match with one step reward'
    assert env._agent_dones == [False for _ in range(env.n_agents)]