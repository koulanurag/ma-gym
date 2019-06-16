# ma-gym
Multi Agent Environments for OpenAI gym

## Installation
```bash
cd ma-gym
pip install -e .
```

## Usage:
```bash
>>> import gym
>>> import ma_gym
>>> env = gym.make('CrossOver-v0')
>>> env.n_agents  # no. of agents
2
>>> env.action_space  # list of action space of each agent
[Discrete(5), Discrete(5)]
>>> env.reset()  # returns list of intial observation of each agent
[[1.0, 0.375, 0.0], [1.0, 0.75, 0.0]]
>>> action_n = env.action_space.sample()  # samples action for each agent
>>> action_n
[0, 3]
>>> obs_n, reward_n, done_n, info = env.step(action_n) 
>>> obs_n
[[1.0, 0.375, 0.01], [1.0, 0.75, 0.01]]  # next observation of each agent
>>> reward_n  # local reward of each agent
[0, 0]
>>> done_n  # terminal flag of each agent
[False, False]
>>> info
{}
>>> episode_terminate = all(done_n)  # episode terminates when all agent die
>>> team_reward = sum(reward_n)  # team reward is simply sum of all local reward
```

## Environments:


| Name  | Description |
| ------------- | ------------- |
| CrossOver-v0  |  Partially Observability  |
| CrossOver-v1  | Full Observability  |

## Demo
![CrossOver](static/gif/CrossOver.gif)
![PredatorPrey](static/gif/PredatorPrey5x5.gif)
