from models_test import Encoder, Decoder, SequencePredictor, DynamicsPredictor

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
dataset_len = 15000
epoch_ct = 50
latent_dims = (30,)
recurrent_dims = (30,)
action_dims = (1,) # TODO: one hot encode
env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
state, info = env.reset()
scene = cv2.resize(env.render(), scene_shape[:2])

"""
Encoder: z_t ~ q_phi(z_t | h_t, x_t)
"""

enc = Encoder(scene_shape, latent_dims[0]).to(device)

"""
Decoder: x_hat_t ~ p_phi(x_hat_t | h_t, z_t)
"""

dec = Decoder(recurrent_dims[0], latent_dims[0]).to(device)

"""
Sequence Model: h_t = f_phi(h_t-1, z_t-1, a_t-1)
"""

sequence_mdl = SequencePredictor(recurrent_dims[0], latent_dims[0], action_dims[0]).to(device)

"""
Dynamics Predictor: z_hat_t ~ p_phi(z_hat_t | h_t)
"""

dynamics_mdl = DynamicsPredictor(latent_dims[0], recurrent_dims[0]).to(device)


opt_enc = optim.AdamW(enc.parameters(), lr=1e-4, amsgrad=True)
opt_dec = optim.AdamW(dec.parameters(), lr=1e-4, amsgrad=True)
opt_seq = optim.AdamW(sequence_mdl.parameters(), lr=1e-4, amsgrad=True)
opt_dyn = optim.AdamW(dynamics_mdl.parameters(), lr=1e-4, amsgrad=True)

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

"""
L_pred(phi) = -ln(p_phi(x_t | z_t, h_t)) - ln(p_phi(r_t | z_t, h_t)) - ln(p_phi(c_t | z_t, h_t))
L_dyn(phi) = max(1, KL[sg(q_phi(z_t | h_t, x_t)) || p_phi(z_t | h_t)])
L_rep(phi) = max(1, KL[q_phi(z_t | h_t, x_t) || sg(p_phi(z_t | h_t))])
"""

# scene_buffer = [element.scene for element in replay_buffer.get()]
replay_buffer.get()

for epoch in range(epoch_ct):
    epoch_losses = 0
    for i in range(batch_size, len(replay_buffer.get()), batch_size):
        batch = replay_buffer.get()[i-batch_size:i]
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_dyn.zero_grad()
        opt_seq.zero_grad()
        x = torch.tensor(np.array([c.scene for c in batch])/255., device=device, dtype=torch.float).permute(0, 3, 1, 2)
        z = enc(x)
        h = torch.zeros((batch_size, recurrent_dims[0])).to(device)
        h_t = torch.zeros(recurrent_dims).to(device).to(device)
        for elem, z_t, c in zip(batch, z.mean, range(len(batch))):
            a_t = torch.Tensor([elem.action]).to(device)
            h[c] = h_t
            # print(h_t.shape, z_t.shape, a_t.shape)
            h_t = sequence_mdl(h_t, z_t, a_t)
            if elem.done:
                h_t = torch.zeros(recurrent_dims).to(device)
            # get hidden states
        x_hat = dec(h, z.mean)

        z_hat = dynamics_mdl(h)

        loss_pred = -x_hat.log_prob(x).sum()

        # max stops gradient backprop if error is small enough
        loss_dyn = torch.max(torch.Tensor(1).to(device), torch.distributions.kl.kl_divergence(torch.distributions.Normal(z.mean.detach(), z.stddev.detach()), z_hat).mean()) # mean and not sum right
        loss_rep = torch.max(torch.Tensor(1).to(device), torch.distributions.kl.kl_divergence(z, torch.distributions.Normal(z_hat.mean.detach(), z_hat.stddev.detach())).mean())

        loss = (B_pred:=1) * loss_pred + (B_dyn:=0.5) * loss_dyn + (B_rep:=0.1) * loss_rep
        loss.backward()
        epoch_losses += loss.item()
        opt_enc.step()
        opt_dec.step()
        opt_seq.step()
        opt_dyn.step()

        x_img = cv2.resize(np.moveaxis((255. * x[0]).cpu().numpy().astype("uint8"), 0, -1), (400, 400))
        x_hat_img = cv2.resize(np.moveaxis((255 * torch.clip(x_hat.mean, 0, 1)[0]).cpu().detach().numpy().astype("uint8"), 0, -1), (400, 400))

        cv2.imshow("Original | AE'd", cv2.hconcat([x_img, x_hat_img]))
        cv2.waitKey(1)

    print("Epoch", epoch, "loss:", epoch_losses)
    # np.random.shuffle(scene_buffer) # TODO: This might be the issue causing part

torch.save(enc, "models/enc.pt")
torch.save(dec, "models/dec.pt")

