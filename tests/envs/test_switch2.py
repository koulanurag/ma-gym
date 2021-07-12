import gym
import pytest
from pytest_cases import parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env():
    env = gym.make('ma_gym:Switch2-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_full():
    env = gym.make('ma_gym:Switch2-v1')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2


def test_reset(env):
    obs_n = env.reset()

    target_obs_n = [[0, 0.17], [0, 0.83]]
    assert env._step_count == 0
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)]
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
    obs_n = env.reset()
    target_obs_n = output
    obs_n, reward_n, done_n, info = env.step(action_n)

    assert env._step_count == 1
    assert env._total_episode_reward == reward_n, 'Total Episode reward doesn\'t match with one step reward'
    assert env._agent_dones == [False for _ in range(env.n_agents)]
    assert obs_n == target_obs_n


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

    assert step_i == env._step_count
    assert env._total_episode_reward == ep_reward
    test_reset(env)


@parametrize_plus('env',
                  [fixture_ref(env),
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


@parametrize_plus('env',
                  [fixture_ref(env),
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


@parametrize_plus('env',
                  [fixture_ref(env),
                   fixture_ref(env_full)])
def test_optimal_rollout(env):
    actions = [[4, 0], [4, 1], [4, 1], [4, 1], [4, 1], [4, 1], [0, 2], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [2, 4]]
    target_rewards = [[-0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1], [-0.1, 5],
                      [-0.1, 0], [-0.1, 0], [-0.1, 0], [-0.1, 0], [-0.1, 0], [5, 0]]
    target_dones = [[False, False], [False, False], [False, False], [False, False], [False, False], [False, False],
                    [False, True], [False, True], [False, True], [False, True], [False, True], [False, True],
                    [True, True]]

    for _ in range(2):  # multiple episodes to ensure it works after reset as well
        env.reset()
        done = [False for _ in range(env.n_agents)]
        step_i = 0
        while not all(done):
            obs, reward_n, done, _ = env.step(actions[step_i])
            assert reward_n == target_rewards[step_i], 'Expected {}, Got {} at step {}'.format(target_rewards[step_i],
                                                                                               reward_n, step_i)
            assert done == target_dones[step_i]
            step_i += 1


@parametrize_plus('env',
                  [fixture_ref(env),
                   fixture_ref(env_full)])
def test_max_steps(env):
    """ All agent remain at their initial position for the entire duration"""
    for _ in range(2):
        env.reset()
        step_i = 0
        done = [False for _ in range(env.n_agents)]
        while not all(done):
            obs, reward_n, done, _ = env.step([4 for _ in range(env.n_agents)])
            target_reward = [env._step_cost for _ in range(env.n_agents)]
            step_i += 1
            assert (reward_n == target_reward), \
                'step_cost is not correct. Expected {} ; Got {}'.format(target_reward, reward_n)
        assert step_i == env._max_steps, 'max-steps should be reached'
