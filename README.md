# Reinforcment-Learning
In this all Projects dealing with reinforcement learning wil be uploaded 
And some of the codes could be derived from the research papers

pre-requisites 

# Install Gym

`pip install gym`

or

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```



An example to see wheter it's working or not


```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

```


* the code for Cartpole environment

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



