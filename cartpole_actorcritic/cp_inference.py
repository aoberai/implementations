# Played around with code from the internet, did not write from scratch

import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import time
from models import Models 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

env = gym.make('CartPole-v1', render_mode="human")
env.reset(seed=340)

model = torch.load('./models/cartpole.pt')

# reset environment and episode reward
state, _ = env.reset()
# state[2] = math.sin(state[2])
ep_reward = 0


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


# for each episode, only run 9999 steps so that we don't
# infinite loop while learning
for t in range(1, 10000):

    # select action from policy
    action = select_action(state)

    # take the action
    state, reward, done, _, _ = env.step(action)

    # state[2] = math.sin(state[2])

    env.render()

    model.rewards.append(reward)
    ep_reward += reward
