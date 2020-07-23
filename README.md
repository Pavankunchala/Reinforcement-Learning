# Reinforcment-Learning
In this all Projects dealing with reinforcement learning wil be uploaded 
And some of the codes could be derived from the research papers

pre-requisites 
# Install Gym

pip install gym

or

git clone https://github.com/openai/gym
cd gym
pip install -e .



An example to see wheter it's working or not



import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

