#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:21:52 2020

@author: pavankunchala
"""


import numpy as np
import matplotlib.pyplot as plt
import gym

import random

env = gym.make('Taxi-v3')

env.render()

Q = {}

for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0



def epsilon_greedy(state,epsilon):
    
    if random.uniform(0,1) < epsilon:
        
        #taking a random action
        
        return env.action_space.sample()
    
    else:
        
        #taking a gredy action
        
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])
    
    
    

alpha = 0.85
gamma = 0.9
epsilon  = 0.8

#performing Sarsa


for i in range(4000):
    
    # we store cumulative reward of each episodes in r
    r = 0
    
    # initialize the state,
    state = env.reset()
    
    # select the action using epsilon-greedy policy
    action = epsilon_greedy(state,epsilon)
    
    while True:
        
       
        env.render()
        
        # then we perform the action and move to the next state, and receive the reward
        nextstate, reward, done, _ = env.step(action)
        
        # again, we select the next action using epsilon greedy policy
        nextaction = epsilon_greedy(nextstate,epsilon) 
    
        # we calculate the Q value of previous state using our update rule
        Q[(state,action)] += alpha * (reward + gamma * Q[(nextstate,nextaction)]-Q[(state,action)])

        # finally we update our state and action with next action and next state
        action = nextaction
        state = nextstate
        
        # store the rewards
        r += reward
        
        # we will break the loop, if we are at the terminal state of the episode
        if done:
            break
            
    print("total reward: ", r)

env.close()
        
    
    

    
    

    
    
 