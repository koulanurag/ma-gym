import gym
import pytest
from pytest_cases import parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_gym:Checkers-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_full():
    env = gym.make('ma_gym:Checkers-v1')
    yield env
    env.close


def test_init(env):
    assert env.n_agents == 2


def test_reset(env):
    import numpy as np
    obs_n = env.reset()

    # add agent 1 obs
    agent_1_obs = [0.0, 0.86]
    agent_1_obs += np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                             [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                             [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]).flatten().tolist()
    # add agent 2 obs
    agent_2_obs = [0.67, 0.86]
    agent_2_obs += np.array([[[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                             [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]).flatten().tolist()

    init_obs_n = [agent_1_obs, agent_2_obs]

    assert env._step_count == 0
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)]
    assert env._agent_dones == [False for _ in range(env.n_agents)]

    for i in range(env.n_agents):
        assert obs_n[i] == init_obs_n[i], \
            'Agent {} observation mis-match'.format(i + 1)


@pytest.mark.parametrize('pos,valid',
                         [((-1, -1), False), ((-1, 0), False), ((-1, 8), False), ((3, 8), False)])
def test_pos_validity(env, pos, valid):
    assert env.is_valid(pos) == valid


@pytest.mark.parametrize('action_n,output',
                         [([1, 1],  # action
                           ([[0.0, 0.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0.67, 0.71, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                            {'lemon': 7, 'apple': 9}))])  # food_count
def test_step(env, action_n, output):
    env.reset()
    target_obs_n, food_count = output
    obs_n, reward_n, done_n, info = env.step(action_n)

    assert obs_n == target_obs_n, 'observation does not match . Expected {}. Got {}'.format(target_obs_n, obs_n)
    for k, v in food_count.items():
        assert info['food_count'][k] == food_count[k], '{} does not match'.format(k)
    assert env._step_count == 1
    assert env._total_episode_reward == reward_n, 'Total Episode reward doesn\'t match with one step reward'
    assert env._agent_dones == [False for _ in range(env.n_agents)]


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


@parametrize_plus('env', [fixture_ref(env)])
def test_rollout_env(env):
    actions = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
               [0, 4], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4]]
    target_rewards = [[-10.01, -1.01], [9.99, 0.99], [-10.01, -1.01], [9.99, 0.99],
                      [-10.01, -1.01], [9.99, 0.99], [-0.01, -0.01], [-0.01, -0.01],
                      [-10.01, -0.01], [9.99, -0.01], [-10.01, -0.01], [9.99, -0.01],
                      [-10.01, -0.01], [9.99, -0.01]]

    for episode_i in range(1):  # multiple episode to validate the seq. again on reset.

        obs = env.reset()
        done = [False for _ in range(env.n_agents)]
        for step_i in range(len(actions)):
            obs, reward_n, done, _ = env.step(actions[step_i])
            assert reward_n == target_rewards[step_i]
            step_i += 1

        assert done == [True for _m in range(env.n_agents)]


@parametrize_plus('env', [fixture_ref(env),
                          fixture_ref(env_full)])
def test_max_steps(env):
    for episode_i in range(2):
        env.reset()
        done = [False for _ in range(env.n_agents)]
        step_i = 0
        while not all(done):
            obs, reward_n, done, _ = env.step([4, 4])
            step_i += 1
        assert step_i == env._max_steps
        assert done == [True for _m in range(env.n_agents)]


@parametrize_plus('env', [fixture_ref(env),
                          fixture_ref(env_full)])
def test_collision(env):
    for episode_i in range(2):
        env.reset()
        obs_1, reward_n, done, _ = env.step([0, 2])
        obs_2, reward_n, done, _ = env.step([0, 2])

        assert obs_1 == obs_2


@parametrize_plus('env', [fixture_ref(env),
                          fixture_ref(env_full)])
def test_revisit_fruit_cell(env):
    for episode_i in range(2):
        env.reset()
        obs_1, reward_1, done, _ = env.step([1, 1])
        obs_2, reward_2, done, _ = env.step([3, 3])
        obs_3, reward_3, done, _ = env.step([1, 1])

        assert reward_1 != reward_3
