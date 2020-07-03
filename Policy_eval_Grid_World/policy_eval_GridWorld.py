#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:42:03 2020

@author: pavankunchala
"""


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import random


#Parameters

gamma = 1 #Discounting Rate range from (0 to 1)
rewardSize = -1
gridSize = 4
terminationStates = [[0,0], [gridSize-1,gridSize-1]]
actions = [[-1,0],[1,0],[0,1],[0,-1]]
numIterations = 1000

#Utilites

def actionRewardFunction(initalPostion,action):
    
    if initalPostion in terminationStates:
        return initalPostion,0
    reward = rewardSize
    finalPosition = np.array(initalPostion) + np.array(action)
    
    
    if -1 in finalPosition or 4 in finalPosition:
        finalPosition = initalPostion
        
    return finalPosition,reward



#initalization
    
valueMap = np.zeros((gridSize,gridSize))


states =  [[i,j] for  i in range(gridSize) for j in range(gridSize)]



#policiy evaluation

deltas = []
for it in range(numIterations):
    copyValueMap = np.copy(valueMap)
    deltaState = []
    for state in states:
        weightedRewards = 0
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action)
            weightedRewards += (1/len(actions))*(reward+(gamma*valueMap[finalPosition[0], finalPosition[1]]))
        deltaState.append(np.abs(copyValueMap[state[0], state[1]]-weightedRewards))
        copyValueMap[state[0], state[1]] = weightedRewards
    deltas.append(deltaState)
    valueMap = copyValueMap
    if it in [0,1,2,9, 99, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")
                



plt.figure(figsize=(20, 10))
plt.legend()
plt.plot(deltas)













