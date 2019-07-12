import gym
import pytest


@pytest.fixture(scope='module')
def env():
    env = gym.make('Checkers-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2


def test_reset(env):
    import numpy as np
    obs_n = env.reset()

    # add agent 1 obs
    agent_1_obs = [0.0, 0.86]
    agent_1_obs += np.array([[0, 0, 0],
                             [1, 3, 0],
                             [2, 0, 0]]).flatten().tolist()
    # add agent 2 obs
    agent_2_obs = [0.67, 0.86]
    agent_2_obs += np.array([[2, 0, 0],
                             [1, 3, 0],
                             [0, 0, 0]]).flatten().tolist()

    init_obs_n = [agent_1_obs, agent_2_obs]

    assert env._step_count == 0
    assert env._total_episode_reward == 0
    assert env._agent_dones == [False for _ in range(env.n_agents)]

    for i in range(env.n_agents):
        assert obs_n[i] == init_obs_n[i], \
            'Agent {} observation mis-match'.format(i + 1)


@pytest.mark.parametrize('pos,valid',
                         [((-1, -1), False), ((-1, 0), False), ((-1, 8), False), ((3, 8), False)])
def test_is_valid(env, pos, valid):
    assert env.is_valid(pos) == valid


@pytest.mark.parametrize('action_n,output',
                         [([1, 1], {'lemon': 7, 'apple': 9})])
def test_step(env, action_n, output):
    env.reset()
    food_count = output
    obs_n, reward_n, done_n, info = env.step(action_n)

    for k, v in food_count.items():
        assert info['food_count'][k] == food_count[k], '{} does not match'.format(k)
    assert env._step_count == 1
    assert env._total_episode_reward == sum(reward_n), 'Total Episode reward doesn\'t match with one step reward'
    assert env._agent_dones == [False for _ in range(env.n_agents)]
