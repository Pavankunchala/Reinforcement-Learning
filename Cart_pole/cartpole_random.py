#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:17:45 2020

@author: pavankunchala
"""

import gym

if __name__ == "__main__":
    
    env = gym.make("CartPole-v0")
    total_reward= 0.0
    total_steps = 0
    obs = env.reset()
    
    while True:
        action = env.action_space.sample()
        obs,reward,done,_ = env.step(action)
        total_reward += reward
        total_steps+=1
        env.render()
        
        if done:
            break
        print("Episode done in %d steps, total reward %.2f" % ( total_steps, total_reward))
        
        
    