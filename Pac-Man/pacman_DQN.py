#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:26:40 2020

@author: pavankunchala
"""

import gym
import time
import numpy as np

import tensorflow as tf
import keras
#creating the environment
env=gym.make('MsPacman-v0')

print(env.action_space)

print(env.observation_space)

n_height = 210
n_width = 160
n_depth = 3
n_shape = [n_height,n_width,n_depth]
n_inputs = n_height * n_width * n_depth
env.frameskip = 3


frame_time = 1.0 / 15 # seconds

n_episodes = 500
scores = []
for i_episode in range(n_episodes):
    t=0
    score=0
    then = 0
    done = False
    env.reset()
    while not done:
        now = time.time()
        if frame_time < now - then:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward
            #env.render()
            then  = now
            t=t+1
    scores.append(score)
            
print('Average score {}, max {}, min {}'.format(np.mean(scores),np.max(scores),np.min(scores) ))

tf.reset_default_graph()
keras.backend.clear_session()

#Applying deep - Q - learning

def policy_q_nn(obs,env):
    
    #explore strategy
    if np.random.random() < explore_rate:
        
        action = env.action_space.sample()
    #exploitation strategy
    else :
        action = np.argmax(q_nn.predict(np.array([obs])))
    return action

def episode(env, policy, r_max = 0,t_max = 0):
    
    # create the empty list to contain game memory
    memory = deque(maxlen=1000)
    
    obs = env.reset()
    state_prev = obs
    
    episode_reward = 0
    done = False
    t = 0
    
    while not done:
        
        action = policy(state_prev, env)
        obs, reward, done, info = env.step(action)
        state_next = obs
        
        # add the state_prev, action, reward, state_new, done to memory
        memory.append([state_prev,action,reward,state_next,done])
        
        # Generate and update the q_values with 
        # maximum future rewards using bellman function:
        states = np.array([x[0] for x in memory])
        states_next = np.array([np.zeros(n_shape) if x[4] else x[3] for x in memory])
        
        
        q_values = q_nn.predict(states)
        q_values_next = q_nn.predict(states_next)
        
        for i in range(len(memory)):
            
            state_prev,action,reward,state_next,done = memory[i]
            if done:
                
                q_values[i,action] = reward
                
            else:
                
                best_q = np.amax(q_values_next[i])
                
                bellman_q = reward + discount_rate * best_q
                q_values[i,action] = bellman_q
                
        # train the q_nn with states and q_values, same as updating the q_table
        q_nn.fit(states,q_values,epochs=1,batch_size=50,verbose=0)
    
        state_prev = state_next
        
        episode_reward += reward
        if r_max > 0 and episode_reward > r_max:
            break
        t+=1
        if t_max > 0 and t == t_max:
            break
    return episode_reward
# experiment collect observations and rewards for each episode
def experiment(env, policy, n_episodes,r_max=0, t_max=0):
    
    rewards=np.empty(shape=[n_episodes])
    for i in range(n_episodes):
        val = episode(env, policy, r_max, t_max)
        #print('episode:{}, reward {}'.format(i,val))
        rewards[i]=val
            
    print('Policy:{}, Min reward:{}, Max reward:{}, Average reward:{}'
        .format(policy.__name__,
              np.min(rewards),
              np.max(rewards),
              np.mean(rewards)))
    
    
from collections import deque 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# build the Q-Network
model = Sequential()
model.add(Flatten(input_shape = n_shape))
model.add(Dense(512, activation='relu',name='hidden1'))
model.add(Dense(9, activation='softmax', name='output'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()
q_nn = model

# Hyperparameters

discount_rate = 0.9
explore_rate = 0.2
n_episodes = 1

# create the empty list to contain game memory
memory = deque(maxlen=1000)

experiment(env, policy_q_nn, n_episodes)


# Hyperparameters

discount_rate = 0.9
explore_rate = 0.2
n_episodes = 100

# create the empty list to contain game memory
memory = deque(maxlen=1000)

experiment(env, policy_q_nn, n_episodes)


from collections import deque 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# build the CNN Q-Network
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), 
                 strides=(1, 1),
                 activation='relu',
                 input_shape=n_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu',name='hidden1'))
model.add(Dense(9, activation='softmax', name='output'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()
q_nn = model


# Hyperparameters

discount_rate = 0.9
explore_rate = 0.2
n_episodes = 100

# create the empty list to contain game memory
memory = deque(maxlen=1000)

experiment(env, policy_q_nn, n_episodes)

env.close()

        

          
        
    

        
            
            
            
            
            
            
            
            
            
            