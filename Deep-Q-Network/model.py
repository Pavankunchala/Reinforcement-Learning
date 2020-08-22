#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:53:55 2020

@author: pavankunchala
"""

import torch

import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    
    def __init__(self,state_size,action_size,seed,fc1_units = 64,fc2_units = 64):
        
        super(QNetwork,self).__init__()
        self.seed= torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_units) # number of nodes hidden in first hidden layer
        self.fc2 = nn.Linear(fc1_units,fc2_units)  # number of nodes hidden in first hidden layer
        self.fc3  =  nn.Linear(fc2_units,action_size)
        
    def forward(self,state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
        
        
        
        
            