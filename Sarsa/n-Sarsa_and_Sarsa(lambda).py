#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:28:44 2020

@author: pavankunchala
"""


# Importing part

import gym
import itertools
import matplotlib
import numpy as np
import pandas as  pd
import sys
import time
import timeit

from collections import namedtuple

import os

import glob

from lib.tile_coding import IHT, tiles

from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.style.use('ggplot')

import io
import base64

from IPython.display import HTML

#creating the environment 

env = gym.make('MountainCar-v0')

env._max_episode_steps = 3000 #increse the upper time limit
np.random.seed(6)  # Make plots reproducible



class QEstimator():
    
    
    def __init__(self,step_size,num_tilings = 8,max_size = 4096,tiling_dim = None, trace = False):
        
        
        self.trace  = trace
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = tiling_dim or num_tilings
        
        #alpha is the fraction of step_size and num_tilings
        
        self.alpha = step_size/num_tilings
        
        #initalzinf the hash table for tile coding and keeping it in max
        self.iht = IHT(max_size)
        
        
        #initalzizinf the weights
        self.weights = np.zeros(max_size)
        if self.trace:
            self.z = np.zeros(max_size)
            
        
        #tilecoding software  partitions  at integer boundaries
        
        
        self.postion_scale  = self.tiling_dim / (env.observation_space.high[0] 
                                          - env.observation_space.low[0] )
        self.velocity_scale = self.tiling_dim/ ( env.observation_space.high[1]
                                                - env.observation_space.low[1] ) 
            
        
    def featurize_state_action(self,state,action):
        
        #returns the featurized repesentation of state action pair
        
        
        featurized = tiles(self.iht,self.num_tilings,
                           [self.postion_scale * state[0],
                           self.velocity_scale * state[1]],
                           [action]      
                           )
        
        return featurized
    
    def predict(self,s , a = None):
        
        #predicitng q-value(s)
        
        
        if a is None:
            features = [self.featurize_state_action(s, i) for  i in 
                        range(env.action_space.n)]
            
        else:
            features = [ self.featurize_state_action(s, a)]
            
        return [np.sum(self.weights[f]) for f in features]
    
    def update(self,s,a,target):
    
        # updates the estimator parameters
        
        features = self.featurize_state_action(s, a)
        
        # linear function Approx
        estimation = np.sum(self.weights[features])
        
        delta = (target-estimation)
        
        if self.trace:
            # self.z[features] += 1  # Accumulating trace
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            self.weights[features] += self.alpha * delta
            
            
    def reset(self,z_only = False):
        
        
        if z_only:
            
            assert self.trace #'q-value estimator has no z to reset.'m
            self.z = np.zeros(self.max_size)
        else:
            if self.trace:
                self.z = np.zeros(self.max_size)
            self.weights = np.zeros(self.max_size)
            
    
    
def make_epsilon_greedy_policy(estimator,epsilon,num_actions ) :
    
    def policy_fn(observation):
        
        action_probs = np.ones(num_actions,dtype = float)*epsilon / num_actions
        
        q_values= estimator.predict(observation)
        
        best_action_idx = np.argmax(q_values)
        
        action_probs[best_action_idx] += (1.0- epsilon )
        
        return action_probs
    return policy_fn



# defining Sarsa n



def sarsa_n(n,env,estimator,gamma = 1.0,epsilon= 1.0):
    
    
    # create epslion greedy policy
    
    policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)
    
    #Resetting the environment
    
    state = env.reset()
    
    action_probs = policy(state)
    
    action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
    
    
    #setting up the stuff
    
    states = [state]
    actions = [ action]
    rewards = [0.0]
    
    #stepping through epsiodes
    
    T = float('inf')
    
    for t in itertools.count():
        
        if t<T:
            
            
            #take a step
            
            next_state, reward,done,_  =env.step(action)
            
            states.append(next_state)
            
            rewards.append(reward)
            
            
            
            if done:
                T  = t+1
                
            else:
                
                #take next step
                
                next_action_probs = policy(next_state)
                
                next_action = np.random.choice(np.arange(len(next_action_probs)),
                                               p = next_action_probs)
                
                actions.append(next_action)
                
                
        update_time = t+1 -n
        
        
        
        if update_time >= 0:  
            
            # Build target
            target = 0
            for i in range(update_time + 1, min(T, update_time + n) + 1):
                target += np.power(gamma, i - update_time - 1) * rewards[i]
            if update_time + n < T:
                q_values_next = estimator.predict(states[update_time + n])
                target += q_values_next[actions[update_time + n]]
            
            # Update step
            estimator.update(states[update_time], actions[update_time], target)
        
        if update_time == T - 1:
            break

        state = next_state
        action = next_action
    
    ret = np.sum(rewards)
    
    return t, ret



def sarsa_lambda(lmbda, env, estimator, gamma=1.0, epsilon=0):
    
   
    
    # Reset the eligibility trace
    estimator.reset(z_only=True)

    # Create epsilon-greedy policy
    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)

    # Reset the environment and pick the first action
    state = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    ret = 0
    # Step through episode
    for t in itertools.count():
        # Take a step
        next_state, reward, done, _ = env.step(action)
        ret += reward

        if done:
            target = reward
            estimator.update(state, action, target)
            break

        else:
            # Take next step
            next_action_probs = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probs)), p=next_action_probs)

            # Estimate q-value at next state-action
            q_new = estimator.predict(
                next_state, next_action)[0]
            target = reward + gamma * q_new
            # Update step
            estimator.update(state, action, target)
            estimator.z *= gamma * lmbda

        state = next_state
        action = next_action    
    
    return t, ret
    


        
        
        
# plotting stuff

def plot_cost_to_go(env, estimator, num_partitions=50):
   
    
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_partitions)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_partitions)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(
        lambda obs: -np.max(estimator.predict(obs)), 2, np.stack([X, Y], axis=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=0, vmax=200)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title("\"Cost To Go\" Function")
    fig.colorbar(p)
    plt.show()  
    
    

def generate_greedy_policy_animation(env, estimator, save_dir):
    """
    Follows (deterministic) greedy policy
    with respect to the given q-value estimator
    and saves animation using openAI gym's Monitor 
    wrapper. Monitor will throw an error if monitor 
    files already exist in save_dir so use unique
    save_dir for each call.
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        env = gym.wrappers.Monitor(
            env, save_dir, video_callable=lambda episode_id: True)
    except gym.error.Error as e:
        print(e.what())

    # Set epsilon to zero to follow greedy policy
    policy = make_epsilon_greedy_policy(
        estimator=estimator, epsilon=0, num_actions=env.action_space.n)
    # Reset the environment
    state = env.reset()
    for t in itertools.count():
        time.sleep(0.01)  # Slow down animation
        action_probs = policy(state)  # Compute action-values
        [action] = np.nonzero(action_probs)[0]  # Greedy action
        state, _, done, _ = env.step(action)  # Take step
        env.render()  # Animate
        if done:
            print('Solved in {} steps'.format(t))
            break
                
                
                
def display_animation(filepath):
    """ Displays mp4 animation in Jupyter."""
    
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))            
     
    
    
def plot_learning_curves(stats, smoothing_window=10):
    """
    Plots the number of steps taken by the agent
    to solve the task as a function of episode number,
    smoothed over the last smoothing_window episodes. 
    """
    
    plt.figure(figsize=(10,5))
    for algo_stats in stats:
        steps_per_episode = pd.Series(algo_stats.steps).rolling(
            smoothing_window).mean()  # smooth
        plt.plot(steps_per_episode, label=algo_stats.algorithm)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode")
    plt.legend()
    plt.show()  
        
            

def plot_grid_search(stats, truncate_steps=400):
    """ 
    Plots average number of steps taken by the agent 
    to solve the task for each combination of
    step size and boostrapping parameter
    (n or lambda).
    """
    # Truncate high step values for clearer plotting
    stats.steps[stats.steps > truncate_steps] = truncate_steps
    
    # We use -1 step values indicate corresponding combination of
    # parameters doesn't converge. Set these to truncate_steps for plotting.
    stats.steps[stats.steps == -1] = truncate_steps
    
    plt.figure()
    for b_idx in range(len(stats.bootstrappings)):
        plt.plot(stats.step_sizes, stats.steps[b_idx, :], 
            label='Bootstrapping: {}'.format(stats.bootstrappings[b_idx]))
    plt.xlabel('Step size (alpha * number of tilings)')
    plt.ylabel('Average steps per episode')
    plt.title('Grid Search {}'.format(stats.algorithm))
    plt.ylim(140, truncate_steps - 100)
    plt.legend()            
        
            
    
            
        
RunStats = namedtuple('RunStats', ['algorithm', 'steps', 'returns'])     
        
    
def run(algorithm, num_episodes=500, **algorithm_kwargs):
    
    """
    Runs algorithm over multilple episodes and logs
    for each episode the complete return (G_t) and the
    number of steps taken.
    """
    
    stats = RunStats(
        algorithm=algorithm, 
        steps=np.zeros(num_episodes), 
        returns=np.zeros(num_episodes))
    
    algorithm_fn = globals()[algorithm]
    
    for i in range(num_episodes):
        episode_steps, episode_return = algorithm_fn(**algorithm_kwargs)
        stats.steps[i] = episode_steps
        stats.returns[i] = episode_return
        sys.stdout.flush()
        print("\rEpisode {}/{} Return {}".format(
            i + 1, num_episodes, episode_return), end="")
    return stats
      
   
     
        

step_size = 0.5  # Fraction of the way we want to move towards target
n = 4  # Level of bootstrapping (set to intermediate value)
num_episodes = 500

estimator_n = QEstimator(step_size=step_size)

start_time = timeit.default_timer()
run_stats_n = run('sarsa_n', num_episodes, n=n, env=env, estimator=estimator_n)
elapsed_time = timeit.default_timer() - start_time

plot_cost_to_go(env, estimator_n)
print('{} episodes completed in {:.2f}s'.format(num_episodes, elapsed_time))
        
  
                 

# Animate learned policy
save_dir='./animations/n-step_sarsa/'
generate_greedy_policy_animation(env, estimator_n, save_dir=save_dir)
[filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
display_animation(filepath)


step_size = 0.5 # Fraction of the way we want to move towards target
lmbda = 0.92  # Level of bootstrapping (set to intermediate value)
num_episodes = 500

estimator_lambda = QEstimator(step_size=step_size, trace=True)

start_time = timeit.default_timer()
run_stats_lambda = run('sarsa_lambda', num_episodes, lmbda=lmbda, env=env, estimator=estimator_lambda)
elapsed_time = timeit.default_timer() - start_time

plot_cost_to_go(env, estimator_lambda)
print('{} episodes completed in {:.2f}s'.format(num_episodes, elapsed_time))

# Animate learned policy
save_dir='./animations/sarsa_lambda/'
generate_greedy_policy_animation(env, estimator_lambda, save_dir=save_dir)
[filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
display_animation(filepath)


plot_learning_curves([run_stats_n, run_stats_lambda])



# comparing

GridSearchStats = namedtuple('GridSearchStats', ['algorithm', 'steps', 'step_sizes', 'bootstrappings'])


def run_grid_search(algorithm, step_sizes, bootstrappings, episodes=100, num_runs=5,
                   **algorithm_kwargs):
    
   
    
    stats = GridSearchStats(
        algorithm=algorithm, 
        steps=np.zeros((len(bootstrappings), len(step_sizes))),
        step_sizes=step_sizes,
        bootstrappings=bootstrappings)
        
    algorithm_fn = globals()[algorithm]
    trace = True if algorithm == 'sarsa_lambda' else False

    for run_idx in range(num_runs):
        for b_idx, bootstrapping in enumerate(bootstrappings):
            for s_idx, step_size in enumerate(step_sizes):
                if algorithm == 'sarsa_n':
                    if (bootstrapping == 8 and step_size > 1) or \
                    (bootstrapping == 16 and step_size > 0.75):
                        # sarsa_n doesn't converge in these cases so 
                        # assign a default value and skip over.
                        stats.steps[b_idx, s_idx] = -1 * num_runs * episodes
                        continue
                estimator = QEstimator(step_size=step_size, trace=trace)
                for episode in range(episodes):
                    sys.stdout.flush()
                    print('\r run: {}, step_size: {}, bootstrapping: {}, episode: {}'.format(
                            run_idx, step_size, bootstrapping, episode), end="")
                    episode_steps, _ = algorithm_fn(
                        bootstrapping, estimator=estimator, **algorithm_kwargs)
                    stats.steps[b_idx, s_idx] += episode_steps
                    
    
    # Average over independent runs and episodes
    stats.steps[:] /= (num_runs * episodes)
   
    return stats


step_sizes = np.arange(0.1, 1.8, 0.1)
ns = np.power(2, np.arange(0, 5))
grid_search_stats_n = run_grid_search('sarsa_n', step_sizes, ns, env=env)
plot_grid_search(grid_search_stats_n)

step_sizes = np.arange(0.1, 1.8, 0.1)
lambdas = np.array([0, 0.68, 0.84, 0.92, 0.98, 0.99])
grid_search_stats_lambda = run_grid_search('sarsa_lambda', step_sizes, lambdas, env=env)
plot_grid_search(grid_search_stats_lambda)







