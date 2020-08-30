# Reinforcment-Learning
**Reinforcement learning (RL)** is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

In this repository we are going have codes for the algorithms of reinforcement learning

* You can also check the instructions to installation of **Gym** [here](https://gym.openai.com/docs/)


## Install Gym

`pip install gym`

or

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

### Example

An example to see wheter _gym_ is working or not


```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

```

### The code for Cartpole environment

```
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

## Table of Contents 
* [Temporal-Difference](https://github.com/Pavankunchala/Reinforcement-Learning/tree/master/Temporal-Difference)
* [K-Armed-Bandit](https://github.com/Pavankunchala/Reinforcement-Learning/tree/master/K-armed-Bandit)
*[Q-learning](https://github.com/Pavankunchala/Reinforcement-Learning/tree/master/Q-Learning)
*[Tile-Coding](https://github.com/Pavankunchala/Reinforcement-Learning/tree/master/Tile-coding%20)




