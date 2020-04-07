#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:20:32 2020

@author: marcus
"""

#environment output at each step: observation, reward, done, info 

#Observations:
    # 0 cart position
    # 1 cart velocity
    # 2 pole angle 
    # 3 pole velocity at tip
#Actions
    # 0 push cart to left
    # 1 push cart to right


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import namedtuple
import matplotlib.pyplot as plt

#Hyperparameters
MEMORY_SIZE = 5000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
GAMMA = 0.999
TARGET_UPDATE = 10
BATCH_SIZE = 128
NUM_EPISODES = 100

Sample = namedtuple('Sample', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    '''
    Memory to save state transistions for batch training.
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ControlNet(nn.Module):
    def __init__(self):
        super(ControlNet, self).__init__()
        layers = []
        layers += [nn.Linear(4,64)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(64,64)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(64,2)]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class Trainer():
    '''
    Class encapsulating the reinforcement learning process. 
    Keeps a policy network and a target network for stability.
    Also initializes the optimizer of choice and the replay memory.
    Keeps track of the number of select_action steps done to adapt the selection procedure.
    '''
    def __init__(self, env, network, memory_size=MEMORY_SIZE, lr=0.01, weight_decay=0.0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        
        self.policy_net = network
        self.policy_net.to(self.device)
        
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.to(device)
        
        self.reward_log = []
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(memory_size)
        
        self.steps_done = 0
        self.rand_actions = []
        self.net_actions = []
        self.total_rand_actions = 0
        self.total_net_actions = 0
        
    def select_action(self, observation):
        '''
        Selects what action to take. Randomly chooses between to ways of action selection:
            1.) Random sampling.
            2.) Inference of the policy_net.
        In further progression of training the random choice will favor method 2.
        '''
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if np.random.rand() > eps_threshold:
            with torch.no_grad():
                #unsqueeze to be able to cat later on
                action = self.policy_net(torch.tensor(observation, dtype=torch.float, device=self.device)).argmax().unsqueeze(0)
                self.total_net_actions += 1
                
        else:
            action = torch.randint(low=0,high=2,size=(1,),device=device, dtype=torch.long)
            self.total_rand_actions += 1
            
        self.net_actions.append(self.total_net_actions)
        self.rand_actions.append(self.total_rand_actions)
        return action
        
        
    def update(self):
        '''
        Training procedure run after every state transition. A train batch is sampled 
         from memory.

        '''
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        
        #invert batch
        batch = Sample(*zip(*batch))
        
        #make a mask for those states where the next_state  is not done
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s, device=self.device) for s in batch.next_state if s is not None]).float()
         
        state_batch = torch.stack(batch.state).float()
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        policy_output = self.policy_net(state_batch)
        state_action_values = policy_output.gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        #Compute Q-values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        #loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradient clipping because RMSProp is used.
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
    def train(self, num_episodes):
        self.reward_log = []
        for i_episode in range(num_episodes):
            # Initialize the environment
            observation = self.env.reset()
            sum_reward = 0
            while(1):
                state = torch.tensor(observation, device=self.device)
                action = self.select_action(observation)
                observation, reward, done, _ = self.env.step(action.item())
                sum_reward += reward
                reward = torch.tensor([reward], device=device)
                
                if done:
                      next_state = None
                else:
                    next_state = observation
                     
                sample = Sample(state, action, next_state, reward)
                self.memory.push(sample)
                
                trainer.update()
                
                
                if done:
                    print("Episode {} done. Reward: {}".format(i_episode, sum_reward))
                    self.reward_log.append(sum_reward)
                    break
                
            if i_episode % TARGET_UPDATE == 0:
                trainer.target_net.load_state_dict(trainer.policy_net.state_dict())
    
    def plot_rewards(self):
        plt.plot(self.reward_log)
        
    def plot_actions(self):
        plt.plot(self.rand_actions, label='random')
        plt.plot(self.net_actions, label='network')
        plt.legend()

    

def test(env, network):
    observation = env.reset()
    
    sum_reward = 0
    i = 0
    while(i<500):
        env.render()
        action = network(torch.tensor(observation, dtype=torch.float, device=device)).argmax().unsqueeze(0)
        observation, reward, done, _ = env.step(action.item())
        sum_reward += reward
        i+=1
        
        # if(done and i<200):
        #      break
    print("Testrun. Reward: {}".format(sum_reward))
    env.close()
#%%




if __name__ =="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v0')
    network = ControlNet()
    trainer = Trainer(env, network, lr=0.01, weight_decay=1e-3)
    trainer.train(NUM_EPISODES)
    
    
   
        
    #%%
    # Test procedure
    # network.load_state_dict(torch.load("./cartpole_net.pth"))
    
    test(env, network)
    
    
    
    
    #%%
    # torch.save(network.state_dict(), "./cartpole_net.pth")