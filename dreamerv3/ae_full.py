from models_test import Encoder, Decoder

import torch
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import cv2
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import time
import sys

device = torch.device("cuda")
display_shape = (400, 400, 3)
scene_shape = (75, 75, 3)
batch_size = 64
dataset_len = 10000
epoch_ct = 15
latent_dim = (15,)
env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
state, info = env.reset()
scene = cv2.resize(env.render(), scene_shape[:2])
enc = Encoder(scene_shape, latent_dim[0]).to(device)
dec = Decoder(latent_dim[0], scene_shape).to(device)
opt_enc = optim.AdamW(enc.parameters(), lr=1e-4, amsgrad=True)
opt_dec = optim.AdamW(dec.parameters(), lr=1e-4, amsgrad=True)

class Element:
    def __init__(self, scene, state, action, nxt_scene, nxt_state, reward, done):
        self.scene = scene
        self.state = state
        self.action = action
        self.nxt_scene = nxt_scene
        self.nxt_state = nxt_state
        self.reward = reward
        self.done = done

    def __str__(self, key):
        return '{}, {}'.format(self.scene, self.state, self.action, self.nxt_scene, self.nxt_state, self.reward, self.done)

# Element-wise data: scene, state, action, nxt_scene, nxt_state, rew, term
class ReplayBuffer:
    # capacity due to memory-constraints
    def __init__(self, capacity=1000000):
        self.buffer = []
        self.capacity = capacity

    def add(self, val):
        while len(self.buffer) > self.capacity:
            del self.buffer[0]
        self.buffer.append(val)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def last(self):
        return self.buffer[-1]

    def get(self):
        return self.buffer

replay_buffer = ReplayBuffer()


def get_action(state):
    # exploration policy is random
    return env.action_space.sample()

scene_buffer = []

for _ in range(dataset_len):
    # agent policy that uses the state and info
    action = get_action(state)
    nxt_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    nxt_scene = cv2.resize(env.render(), scene_shape[:2])
    replay_buffer.add(Element(scene, state, action, nxt_scene, nxt_state, reward, done))
    cv2.imshow("Win", cv2.resize(replay_buffer.last().nxt_scene, display_shape[:2]))
    cv2.waitKey(1)

    state = nxt_state
    scene = nxt_scene

    if done:
        state, info = env.reset()



scene_buffer = [element.scene for element in replay_buffer.get()]

for epoch in range(epoch_ct):
    epoch_losses = 0
    for i in range(batch_size, len(scene_buffer), batch_size):
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        x = torch.tensor(np.array(scene_buffer[i-batch_size:i])/255., device=device, dtype=torch.float).permute(0, 3, 1, 2)
        z = enc(x)
        x_hat = dec(z)
        loss = -torch.distributions.Normal(x_hat, 5).log_prob(x).sum()
        # loss = (torch.sum((x - x_hat) ** 2))
        epoch_losses += loss.item()
        loss.backward()
        opt_enc.step()
        opt_dec.step()
        
        print(x_hat.shape)
        x_hat = torch.clip(x_hat, 0, 1)
        x_img = cv2.resize(np.moveaxis((255. * x[0]).cpu().numpy().astype("uint8"), 0, -1), (400, 400))
        x_hat_img = cv2.resize(np.moveaxis((255 * x_hat[0]).cpu().detach().numpy().astype("uint8"), 0, -1), (400, 400))
        # print(np.max(x_hat_img), np.min(x_hat_img))
        # print(torch.max(x_hat), torch.min(x_hat))
        cv2.imshow("Original | AE'd", cv2.hconcat([x_img, x_hat_img]))
        cv2.waitKey(1)

    print("Epoch", epoch, "loss:", epoch_losses)
    np.random.shuffle(scene_buffer)

torch.save(enc, "models/enc.pt")
torch.save(dec, "models/dec.pt")

