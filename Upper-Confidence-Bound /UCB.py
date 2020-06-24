#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:25:26 2020

@author: pavankunchala
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ucb_bandit:
    
    
    def __init__(self, k ,c, iters, mu = 'random'):
        # number of arms 
        
        self.k = k
         # the condindencre bound(exploration parameter)
        self.c = c
        
        # number of iters
        self.iters = iters
        
        #no of times steps
        self.n = 1
        #step count for each arm
        self.k_n  =np.ones(k)
        #mean reward
        self.mean_reward  = 0
        self.reward = np.zeros(iters)
        
        #reward for each arm
        self.k_reward=np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            #user defined average
            self.mu = np.array(mu)
        elif mu == 'random':
            #draws random from the prob distrubtytion
            self.mu = np.random.normal(0,1,k)
        elif mu == 'sequence':
            self.mu = np.linspace(0, k-1, k)
            
    def pull(self):
        
        a = np.argmax(self.k_reward +self.c *np.sqrt(np.log(self.n) / self.k_n))
        
        reward = np.random.normal(self.mu[a],1)
        
        #updte
        self.n +=1
        self.k_n[a]+=1
        
        
        #update total
        self.mean_reward = self.mean_reward +(reward - self.mean_reward)/self.n
        
        self.k_reward[a] = self.k_reward[a] +(reward - self.k_reward[a])/self.k_n[a]
        
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i]= self.mean_reward
            
    def reset(self,mu = 'none'):
        
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)
            
k = 10 # number of arms
iters = 1000
ucb_rewards = np.zeros(iters)
# Initialize bandits
ucb = ucb_bandit(k, 2, iters)
episodes = 1000
# Run experiments
for i in range(episodes): 
    ucb.reset('random')
    # Run experiments
    ucb.run()
    
    # Update long-term averages
    ucb_rewards = ucb_rewards + (
        ucb.reward - ucb_rewards) / (i + 1)
    
plt.figure(figsize=(12,8))
plt.plot(ucb_rewards, label="UCB")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average UCB Rewards after " 
          + str(episodes) + " Episodes")
plt.show()



            
        
        
            
            
        
        