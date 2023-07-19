"""
https://arxiv.org/pdf/2301.04104.pdf

Steps:

Recurrent State Space Model

Sequence Model: h_t = f_theta(h_t-1, z_t-1, a_t-1)
Encoder: z_t ~ q_theta(z_t | h_t, x_t)
Dynamics Predictor: z_hat_t ~ p_theta(z_hat_t | h_t)
Reward & Continue Predictor: r_hat_t, c_hat_t ~ p_theta(r_hat_t & c_hat_t | h_t, z_t)
Decoder: x_hat_t ~ p_theta(x_hat_t | h_t, z_t)


"""

import gymnasium as gym
import torch
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Encoder, Decoder

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
        from IPython import display
plt.ion()

# env = gym.make("CartPole-v1")
env = gym.make("LunarLander-v2")
observation, info = env.reset()
batch_size = 128

scene_shape = (75, 75, 3)
device = torch.device("cuda")
enc, dec = Encoder(scene_shape, latent_dim).to(device), Decoder(latent_dim, scene_shape).to(device)
opt_enc, opt_dec = optim.AdamW(enc.parameters(), lr=1e-4, amsgrad=True), optim.AdamW(dec.parameters(), lr=1e-4, amsgrad=True)

# Element-wise data: scene, state, action, nxt_scene, nxt_state, rew, term
class ReplayBuffer:
    # capacity due to memory-constraints
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity

    def add(self, *val):
        while len(self.buffer) > self.capacity:
            del self.buffer[0]
        self.buffer.append(val)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def last(self):
        return self.buffer[-1]

buffer = ReplayBuffer(buffer_size)

def get_action(state):
    # random policy
    return env.action_space.sample()

for _ in range(15000):
    # agent policy that uses the observation and info
    action = get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    img = env.render()
    scene_buffer.append(cv2.resize(img, (75, 75)))
    cv2.imshow("Win", cv2.resize(scene_buffer[-1], (400, 400)))
    cv2.waitKey(1)


