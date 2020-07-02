#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 08:35:45 2020

@author: pavankunchala

"""
import numpy as np
import matplotlib.pyplot as plt


"""
 in Gradient algortihm we use preferences if the following actions is more prefered the agent is more
 likely to prefer that action we can find the preference using softmax 
 """
# def softmax(x):
#     return np.exp(x - x.max()) /np.sum(np.exp(x - x.max()),axis = 0)


class grad_bandit:
     
    def __init__(self, k, alpha, iters, mu='random'):
        # Number of arms
        self.k = k
        self.actions = np.arange(k)
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Initialize preferences
        self.H = np.zeros(k)
        # Learning rate
        self.alpha = alpha
         
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
             
    def softmax(self):
        self.prob_action = np.exp(self.H - np.max(self.H)) \
            / np.sum(np.exp(self.H - np.max(self.H)), axis=0)
         
    def pull(self):
        # Update probabilities
        self.softmax()
        # Select highest preference action
        a = np.random.choice(self.actions, p=self.prob_action)
             
        reward = np.random.normal(self.mu[a], 1)
         
        # Update counts
        self.n += 1
        self.k_n[a] += 1
         
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
         
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
         
        # Update preferences
        self.H[a] = self.H[a] + \
            self.alpha * (reward - self.mean_reward) * (1 -
                self.prob_action[a])
        actions_not_taken = self.actions!=a
        self.H[actions_not_taken] = self.H[actions_not_taken] - self.alpha * (reward - self.mean_reward) * self.prob_action[actions_not_taken]
             
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
             
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        self.H = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)
            
            
            
            

k = 10
iters = 1000
# Initialize bandits
grad = grad_bandit(k, 0.1, iters, mu='random') 

grad_rewards = np.zeros(iters)
opt_grad = 0

episodes = 1000
# Run experiments
for i in range(episodes):
    # Reset counts and rewards
    grad.reset('random')
    
    
    grad.run()
     
    grad_rewards = grad_rewards + (
        grad.reward - grad_rewards) / (i + 1)
    
    opt_grad += grad.k_n[np.argmax(grad.mu)]
    
    

          
        
        
plt.figure(figsize=(12,8))
plt.plot(grad_rewards, label="Gradient") 

plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average Gradient Bandit Rewards after "
          + str(episodes) + " Episodes")
plt.show()   
        
        
        
        
        

