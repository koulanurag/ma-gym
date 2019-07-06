# ma-gym
Multi Agent Environments for OpenAI gym

## Installation
```bash
cd ma-gym
python setup.py install # use 'develop' instead of 'install' if developing the package
```

## Usage:
```python
import gym
import ma_gym

env = gym.make('CrossOver-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
env.close()
```

Please refer to [Wiki](https://github.com/koulanurag/ma-gym/wiki) for complete usage details

## Environments:
- [x] Checkers
- [x] Combat
- [x] CrossOver
- [ ] Fetch
- [x] PredatorPrey
- [x] Pong Duel  ```(two player pong game)```
- [x] Switch
- [ ] Traffic Junction

```
Note : openai's environment can be accessed in multi agent form by prefix "ma_".Eg: ma_CartPole-v0
This returns an instance of CartPole-v0 wrapped around multi agent from having a single agent. 
These environments are helpful during debugging.
```

Please refer to [Wiki](https://github.com/koulanurag/ma-gym/wiki) for more details

## Zoo!

| __Checkers-v0__ | __Combat-v0__ | __PongDuel-v0__ |
|:---:|:---:|:---:|
|![Checkers-v0.gif](static/gif/Checkers-v0.gif)|![Combat-v0.gif](static/gif/Combat-v0.gif)|![PongDuel-v0.gif](static/gif/PongDuel-v0.gif)|
| __PredatorPrey5x5-v0__ | __PredatorPrey7x7-v0__ | __Switch1-v0__ |
|![PredatorPrey5x5-v0.gif](static/gif/PredatorPrey5x5-v0.gif)|![PredatorPrey7x7-v0.gif](static/gif/PredatorPrey7x7-v0.gif)|![Switch1-v0.gif](static/gif/Switch1-v0.gif)|



