# ma-gym
A collection of multi agent environments based on OpenAI gym.

[![Build Status](https://travis-ci.com/koulanurag/ma-gym.svg?token=DM2fKnVJXEZDaszt9oHm&branch=master)](https://travis-ci.com/koulanurag/ma-gym)

## Installation
Using PyPI:
```bash
pip install ma-gym
```

Directly from source:
```bash
git clone https://github.com/koulanurag/ma-gym.git
cd ma-gym
pip install -e .
```

## Usage:
```python
import gym
import ma_gym

env = gym.make('Switch2-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
env.close()
```

Please refer to [Wiki](https://github.com/koulanurag/ma-gym/wiki/Usage) for complete usage details

## Environments:
- [x] Checkers
- [x] Combat
- [x] PredatorPrey
- [x] Pong Duel  ```(two player pong game)```
- [x] Switch
- [x] Lumberjacks

```
Note : openai's environment can be accessed in multi agent form by prefix "ma_".Eg: ma_CartPole-v0
This returns an instance of CartPole-v0 in "multi agent wrapper" having a single agent. 
These environments are helpful during debugging.
```

Please refer to [Wiki](https://github.com/koulanurag/ma-gym/wiki/Environments) for more details.

## Zoo!

| __Checkers-v0__ | __Combat-v0__ | __Lumberjacks-v0__ |
|:---:|:---:|:---:|
|![Checkers-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/Checkers-v0.gif)|![Combat-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/Combat-v0.gif)|![Lumberjacks-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/Lumberjacks-v0.gif)|
| __PongDuel-v0__ | __PredatorPrey5x5-v0__ | __PredatorPrey7x7-v0__ |
| ![PongDuel-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/PongDuel-v0.gif) | ![PredatorPrey5x5-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/PredatorPrey5x5-v0.gif) | ![PredatorPrey7x7-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/PredatorPrey7x7-v0.gif) |
|                 __Switch2-v0__                 |                        __Switch4-v0__                        |                                                              |
|  ![Switch2-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/Switch2-v0.gif)  |         ![Switch4-v0.gif](https://raw.githubusercontent.com/koulanurag/ma-gym/master/static/gif/Switch4-v0.gif)         |                                                              |

## Testing:

- Install: ```pip install pytest ```
- Run: ```pytest```


## Reference:
Please use this bibtex if you would like to cite it:
```
@misc{magym,
      author = {Koul, Anurag},
      title = {ma-gym: Collection of multi-agent environments based on OpenAI gym.},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/koulanurag/ma-gym}},
    }
```
## Acknowledgement:
This project was developed to complement my research internship @ [SAS](https://www.sas.com/en_us/home.html).


