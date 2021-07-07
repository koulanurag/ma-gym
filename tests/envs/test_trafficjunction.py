import gym
import numpy as np
import pytest
from pytest_cases import pytest_parametrize_plus, fixture_ref


@pytest.fixture(scope='module')
def env_4():
    env = gym.make('ma_gym:TrafficJunction4-v0')
    yield env
    env.close()


@pytest.fixture(scope='module')
def env_10():
    env = gym.make('ma_gym:TrafficJunction10-v0')
    yield env
    env.close()


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4), fixture_ref(env_10)])
def test_init(env):
    assert 1 <= env._n_max <= 10, 'N_max must be between 1 and 10, got {}'.format(env._n_max)
    assert env._on_the_road.count(True) <= len(env._entry_gates), 'Cars on the road after initializing cannot ' \
                                                                  'be higher than {}'.format(len(env._entry_gates))


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4), fixture_ref(env_10)])
def test_reset(env):
    env.reset()
    assert env._step_count == 0, 'Step count should be 0 after reset, got {}'.format(env._step_count)
    assert env._agent_step_count == [0 for _ in range(env.n_agents)], 'Agent step count should be 0 for all agents' \
                                                                      ' after reset'
    assert env._total_episode_reward == [0 for _ in range(env.n_agents)], 'Total reward should be 0 after reset'
    assert env._agent_dones == [False for _ in range(env.n_agents)], 'Agents cannot be done when the environment' \
                                                                     ' resets'
    assert env._agent_turned == [False for _ in range(env.n_agents)], 'Agents cannot have changed direction ' \
                                                                      ' when the environment resets'


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4), fixture_ref(env_10)])
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

    assert step_i == env._step_count, 'Number of steps after an episode must match ' \
                                      'env._step_count, got {} and {}'.format(step_i, env._step_count)
    assert env._total_episode_reward == ep_reward, 'Total reward after an episode must match' \
                                                   ' env._total_episode_reward, ' \
                                                   'got {} and {}'.format(ep_reward, env._total_episode_reward)
    test_reset(env)


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4), fixture_ref(env_10)])
def test_observation_space(env):
    obs = env.reset()
    expected_agent_i_shape = (np.prod(env._agent_view_mask) * (env.n_agents + 2 + 3),)
    for agent_i in range(env.n_agents):
        assert obs[agent_i].shape == expected_agent_i_shape, \
            'shape of obs. expected to be {}; but found to be {}'.format(expected_agent_i_shape, obs[agent_i].shape)

    assert env.observation_space.contains(obs), 'Observation must be part of the observation space'
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        obs, reward_n, done, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), 'Observation must be part of the observation space'
    assert env.observation_space.contains(obs), 'Observation must be part of the observation space'
    assert env.observation_space.contains(env.observation_space.sample()), 'Observation must be part of the' \
                                                                           ' observation space'


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4)])
def test_step_cost_env4(env):
    env.reset()
    for step_i in range(3):  # small number of steps so that no collision occurs
        obs, reward_n, done, _ = env.step(env.action_space.sample())
        target_reward = [env._step_cost * (step_i + 1) for _ in range(env.n_agents)]
        assert (reward_n == target_reward), \
            'step_cost is not correct. Expected {} ; Got {}'.format(target_reward, reward_n)


@pytest_parametrize_plus('env',
                         [fixture_ref(env_10)])
def test_step_cost_env10(env):
    env.reset()
    for step_i in range(1):  # just 1 step so that no collision occurs
        obs, reward_n, done, _ = env.step(env.action_space.sample())
        target_reward = [env._step_cost * (step_i + 1) for _ in range(4)] + [0 for _ in range(env.n_agents - 4)]
        assert (reward_n == target_reward), \
            'step_cost is not correct. Expected {} ; Got {}'.format(target_reward, reward_n)


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4)])
def test_all_brake_rollout_env4(env):
    """ All agent apply brake for the entire duration"""
    for _ in range(2):
        env.reset()
        step_i = 0
        done = [False for _ in range(env.n_agents)]
        while not all(done):  # small number of steps so that no collision occurs
            obs, reward_n, done, _ = env.step([1 for _ in range(env.n_agents)])
            target_reward = [env._step_cost * (step_i + 1) for _ in range(env.n_agents)]
            step_i += 1
            assert (reward_n == target_reward), \
                'step_cost is not correct. Expected {} ; Got {}'.format(target_reward, reward_n)
        assert step_i == env._max_steps, 'max-steps should be reached'


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4)])
def test_one_gas_others_brake_rollout_env4(env):
    """
    "Agent 0" applies gas and others brake. This will mean that there will not be any collision and "Agent 0" will
    reach it's destination in minimal number of steps; beyond which reward for agent "0" would be 0.
    """

    # testing over multiple episode to ensure it works with multiple routes assigned to "agent 0"
    for episode_i in range(5):
        obs = env.reset()
        step_i = 0
        route_max_steps = [13, 12, 14]  # routes [fwd, r, l]
        done = [False for _ in range(env.n_agents)]
        agent_0_route = obs[0].reshape((9, 9))[4][6:]  # one-hot
        agent_0_route_idx = np.where(agent_0_route == 1)[0][0] 
        max_agent_0_steps = route_max_steps[agent_0_route_idx]
        while not all(done):  # small number of steps so that no collision occurs
            obs, reward_n, done, _ = env.step([0] + [1 for _ in range(env.n_agents - 1)])
            target_reward = [env._step_cost * (step_i + 1) for _ in range(env.n_agents)]
            # once the car reaches destination, there is no step cost
            if step_i >= max_agent_0_steps:
                target_reward[0] = 0
            step_i += 1
            assert (reward_n == target_reward), \
                'step_cost is not correct. Expected {} ; Got {}, Episode {} Agent 0 route: {} '.format(target_reward,
                                                                                                      reward_n,
                                                                                                      episode_i,
                                                                                                      agent_0_route)
        assert step_i == env._max_steps, 'max-steps should be reached'


@pytest_parametrize_plus('env',
                         [fixture_ref(env_4)])
def test_all_gas_all_routes_forward_rollout_env4(env):
    """
    All the agents apply gas and follow the forward route (1). This will mean that all the agents 
    will reach the junction at the same time and then will collide in the junction with each other
    in a deadlock and never reach the destination.
    """

    obs = env.reset()
    env._agents_routes = [1 for _ in range(env.n_agents)]  # changes all routes to fwd
    step_i = 0
    done = [False for _ in range(env.n_agents)]
    while not all(done):
        _, reward_n, done, info = env.step([0 for _ in range(env.n_agents)])  # all gas
        target_reward = [env._step_cost * (step_i + 1) for _ in range(env.n_agents)]
        if step_i >= 6:
            target_reward = [target_reward[agent_i] + env._collision_reward for agent_i in range(env.n_agents)]
            assert info['step_collisions'] == 4, 'collision count is not correct. '\
                    'Expected 4 ; Got {}'.format(info['step_collisions'])
        step_i += 1
        assert (reward_n == target_reward), \
            'collision reward is not correct. Expected {} ; Got {} '.format(target_reward, reward_n)
    assert step_i == env._max_steps, 'max-steps should be reached'
