#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:01:39 2020

@author: pavankunchala
"""


#Q learning

import random
import gym

env = gym.make('Taxi-v3')

env.render()


q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s,a)] = 0.0
        

def update_q_table(prev_state,action,reward,next_state,alpha,gamma):
    qa = max([q[(next_state,a)] for a in range(env.action_space.n)])
    q[prev_state,action] += alpha *(reward + qa - q[prev_state,action])
    


def epsilon_greedy(state,epsilon):
    
    if random.uniform(0, 1) < epsilon:
        
        #takinf a random action
        return env.action_space.sample()
    
    else:
        
        #taking a greedy action
        return max(list(range(env.action_space.n)), key = lambda x: q[(state,x)])
    
    



alpha = 0.4
gamma = 0.999
epsilon = 0.017

for i in range(8000):
    r = 0
    prev_state = env.reset()
    #env.render()
    
    
    while True:
        
        env.render()
        
    
        action = epsilon_greedy(prev_state, epsilon)
        
        next_state, reward,done , _ = env.step(action)
        
        update_q_table(prev_state, action, reward, next_state, alpha, gamma)
        
        prev_state  = next_state
        
        r += reward
        
        
        if done:
            break
        
    print("total reward:", r)
    
    
env.close()


        
        
        
        
    
        
    