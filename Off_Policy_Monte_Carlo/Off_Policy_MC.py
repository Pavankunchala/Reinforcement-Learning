#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:14:54 2020

@author: pavankunchala
"""


import gym
import matplotlib.pyplot as plt

import numpy as np

import sys

import collections

from collections import defaultdict

#from lib.envs.blackjack import BlackjackEnv

from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
from matplotlib import cm






env = gym.make('Blackjack-v0')


def create_random_policy(nA):
    
    #create a random policy which takes no of actions and gives prob based on the no of 
    #no of actions
    
    A = np.ones(nA,dtype = float)/nA
    
    def policy_fn(observation):
        return A
    return policy_fn


#creating a greedy policy
    
def create_greedy_policy(Q):
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype = float)
        best_action = np.argmax(Q[state])
        
        A[best_action]  = 1.0
        return A
    return policy_fn


#monte carlo control off policy method using weighted importance sampling 
#for finding greedy optimal polciy
    

def mc_control_importance_sampling(env,num_episodes,behaviour_policy,discount_factor = 1.0):
    
    
    #A dictionary thats maps  state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #the cumulative denominator  of weighted importance sampling
    
    C = defaultdict(lambda:np.zeros(env.action_space.n))
    
    #the greedy polcy we want to learn
    target_policy = create_greedy_policy(Q)
    
    
    for i_episode in range(1,num_episodes+1):
        
        
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        #Generate an episode {it's an array of state state , action ,reward}
            
        episode = []
        state = env.reset()
        
        for t in range(100):
            
            #sampling an action from our policy
            probs = behaviour_policy(state)
            action = np.random.choice(np.arange(len(probs)), p = probs)
            
            next_state, reward , done ,_ = env.step(action)
            episode.append((state,action,reward))
            
            if done:
                break
            state = next_state
            
            
        #Sum of discounted sums
        G = 0.0
        #The importance samplingg ratio
        W = 1.0
        
        
        #for each step in episode,backwards
        for t in range(len(episode))[::-1]:
            
            
            state,action,reward  = episode[t]
            
            # update total reward
            G = discount_factor *G +reward
            
            #updating weight importance sampling 
            C[state][action] +=W
            
            Q[state][action]  += (W/C[state][action])* (G - Q[state][action])
            
            
            if action != np.argmax(target_policy(state)):
                break
            W = W*1/behaviour_policy(state)[action]
            
    return Q, target_policy



random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes = 500000, behaviour_policy= random_policy)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value




            
    
        
        
