#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:50:01 2020

@author: pavankunchala
"""

#importing stuff

import gym
import numpy as np
import math
from collections import deque
import matplotlib.pyplot  as plt

#torch importing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Initalizing the environment

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)

print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)


#Creating the agent

class Agent(nn.Module):
    
    
    def __init__(self,env, h_size = 16):
        super(Agent, self).__init__()
        self.env = env
        
        #state 
        self.s_size = env.observation_space.shape[0]
        #Hidden layer
        self.h_size = h_size
        #actionsize
        self.a_size = env.action_space.shape[0]
        
        #defining the layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size,self.a_size)
        
        
    def set_weights(self, weights):
        
        s_size = self.s_size
        h_size= self.h_size
        a_size = self.a_size
        
        # seprate the weighs for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(weights[: s_size * h_size].reshape(s_size,h_size))
        fc1_b = torch.from_numpy(weights[ s_size* h_size : fc1_end])
        fc2_W  = torch.from_numpy(weights[fc1_end :fc1_end +(h_size * a_size)] . reshape(h_size,a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size) : ])
        
        #set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        
    def get_weights_dim(self):
        return (self.s_size +1) * self.h_size + (self.h_size +1)* self.a_size
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data
    
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return
    
    
agent = Agent(env).to(device)
    
    
def cem(n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    best_weight = sigma*np.random.randn(agent.get_weights_dim())

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    return scores

scores = cem()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()     


agent.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
while True:
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = agent(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break

env.close() 
    
        
        





















