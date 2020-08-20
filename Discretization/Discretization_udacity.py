#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:09:47 2020

@author: pavankunchala
"""


#Importing stuff

import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

#creating the environment

env= gym.make('MountainCar-v0')
env.seed(505)  #Random seeding

#creating a Random agent 
state= env.reset()
score = 0
for t in range(200):
    action = env.action_space.sample()
    #env.render()
    state,reward, done,_ = env.step(action)
    score += reward
    if done:
        break
    
print('Final Score: ',score)
env.close()




# now we have to create a uniform grid in this environment 

def create_uniform_grid(low,high,bins= (10,10)):
   
    grid = [np.linspace(low[dim], high[dim],bins[dim]+1)[1 :-1] for dim in range(len(bins))]

    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid



low = [-1.0, -5.0]
high = [1.0, 5.0]
create_uniform_grid(low, high) # testing

# we are going to discretize

def discretize(sample, gird): 
    return list(int(np.digitize(s,g)) for s ,g in zip(sample,gird)) # applying on each dimension

 # testing
grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
samples = np.array(
    [[-1.0 , -5.0],
     [-0.81, -4.1],
     [-0.8 , -4.0],
     [-0.5 ,  0.0],
     [ 0.2 , -1.9],
     [ 0.8 ,  4.0],
     [ 0.81,  4.1],
     [ 1.0 ,  5.0]])
discretized_samples = np.array([discretize(sample, grid) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")



def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """Visualize original and discretized samples on a given 2-dimensional grid."""

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)
    
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  # add a line connecting each original-discretized sample
    ax.legend(['original', 'discretized'])

    
visualize_samples(samples, discretized_samples, grid, low, high)

#Create a grid to discretize the state space
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
state_grid
# Obtain some samples from the space, discretize them, and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples])
visualize_samples(state_samples, discretized_state_samples, state_grid,
                  env.observation_space.low, env.observation_space.high)
plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space


# now as we are done with the discretization stuff let's get to Q learning


class QLearningAgent:
    
    # we can use this Agent to act on contious space by discretizing it
    
    def __init__(self,env,state_grid,alpha = 0.02,gamma = 0.99, epsilon  = 1.0,
                 epsilon_decay_rate = 0.9995,min_epsilon = .01,seed = 505):
        
        
        # Environment Info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) +1 for  splits in self.state_grid) # n dimendional space
        self.action_size = self.env.action_space.n #dimensional  discrete space size
        self.seed = np.random.seed(seed)
        print(" ")
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        print(" ")
        
        
        #Learning parameters
        self.alpha  = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = self.inital_epsilon  = epsilon #Exploratory factor
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease the epsilon
        self.min_epsilon = epsilon
        
        #creating a Q table
        self.q_table  = np.zeros(shape = (self.state_size +(self.action_size,)))
        
        print("Q table size:", self.q_table.shape)
        print(" ")
        
        
    def preprocess_state(self,state):
        return  tuple(discretize(state, self.state_grid))
    
    def reset_episode(self,state):
        
        #Gradually decreasing the exploratory rate
        
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon,self.min_epsilon)
        
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self,epsilon = None):
        
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon
        
    def act(self,state , reward = None, done = None, mode = 'train'):
        
        state = self.preprocess_state(state)
        
        if mode == 'test':
            
            action = np.argmax(self.q_table[state])
            
        else:
            
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])
                
             # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])
                
        self.last_state = state
        self.last_action = action
        return action
    
q_agent = QLearningAgent(env, state_grid)


#Running the agent

def run(agent,env, num_episodes = 20000,mode = 'train'):
    
    scores = []
    max_avg_score = -np.inf
    
    for i_episode in range(1, num_episodes +1):
        state = env.reset()
        action= agent.reset_episode(state)
        total_reward = 0
        done = False
        
        while not done:
           
            state,reward,done,info = env.step(action)
            total_reward += reward
            action = agent.act(state,reward,done,mode)
            
            
            
        #save final scores
        scores.append(total_reward)
        
        #print episode stats
        
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()
                
    return scores

scores = run(q_agent, env)

# Plot scores obtained per episode
plt.plot(scores); plt.title("Scores");

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean

rolling_mean = plot_scores(scores)

# Run in test mode and analyze scores obtained
test_scores = run(q_agent, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = plot_scores(test_scores)

def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet');
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')


plot_q_table(q_agent.q_table)

state_grid_new = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
q_agent_new = QLearningAgent(env, state_grid_new)
q_agent_new.scores = []

q_agent_new.scores += run(q_agent_new, env, num_episodes=50000)  # accumulate scores
rolling_mean_new = plot_scores(q_agent_new.scores)


plot_q_table(q_agent_new.q_table)

state = env.reset()
score = 0
for t in range(200):
    action = q_agent_new.act(state, mode='test')
    env.render()
    state, reward, done, _ = env.step(action)
    score += reward
    if done:
        break 
print('Final score:', score)
env.close()

            
        
        
    
    





            
            
        
        
        
        
        
        































