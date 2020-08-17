# Reinforcment-Learning
In this all Projects dealing with reinforcement learning wil be uploaded 
And some of the codes could be derived from the research papers

pre-requisites 

![alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fai%25C2%25B3-theory-practice-business%2Freinforcement-learning-part-1-a-brief-introduction-a53a849771cf&psig=AOvVaw3NviVxTp0ZB4Bge1a18ASc&ust=1597756733471000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCLjN7-apousCFQAAAAAdAAAAABAD)
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



