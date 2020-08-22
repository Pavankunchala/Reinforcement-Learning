#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:33:12 2020

@author: pavankunchala
"""

import gym
import numpy as np


# creating the environment

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


# creating a random agent
env.reset()
 
score = 0

for  i in range(500):
    action = env.action_space.sample()
    env.render()
    state,reward,done,info = env.step(action)
    score += reward
    if done:
        break
    
env.close()

    
print("Score is:", score)  