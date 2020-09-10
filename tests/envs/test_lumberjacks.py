import gym
import pytest
import ma_gym
from pytest_cases import parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env():
    env = gym.make('Lumberjacks-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_full():
    env = gym.make('Lumberjacks-v1')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2
    assert env._n_trees == 12


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert len(env._objects_dict) == 2 + 12  # 2 agents + 12 trees
    for agent_id, agent in env._agent_generator():
        assert agent.done == False
        assert agent.reward == 0
    for tree_id, tree in env._tree_generator():
        assert tree.alive == True
        assert 0 < tree.strength <= env.n_agents


@pytest.mark.parametrize('action_n', [[0, 0]])  # no-op action
def test_step(env, action_n):
    env.reset()
    obs_n, reward_n, done_n, info = env.step(action_n)

    assert env._step_count == 1
    for (agent_id, agent), reward in zip(env._agent_generator(), reward_n):
        assert agent.done == False, 'Game cannot finished after one step'
        assert agent.reward == reward, 'Total Episode reward doesn\'t match with one step reward'


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
    for (agent_id, agent), reward in zip(env._agent_generator(), ep_reward):
        assert agent.done == True
        assert agent.reward == reward, 'Total Episode reward doesn\'t match with one step reward'
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
