from models import Encoder, Decoder, SequencePredictor, DynamicsPredictor

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
batch_size = 128
dataset_len = 15000
epoch_ct = 50
latent_dims = (30,)
recurrent_dims = (30,)
project_depth = 16
action_dims = (1,) # TODO: one hot encode
env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

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


first_scene = cv2.resize(env.render(), scene_shape[:2])

cv2.imshow("First", first_scene)
cv2.waitKey(0)

a_0 = torch.Tensor([0]).to(device)
x_0 = torch.Tensor(np.moveaxis(first_scene, -1, 0)).unsqueeze(0).to(device)/255.
h_0 = torch.zeros((1, recurrent_dims[0])).to(device)
z_0 = enc(x_0).mean

h_t = h_0
z_t = z_0

print("Action: left")

for i in range(150):
    h_t = sequence_mdl(h_t.squeeze(), z_t.squeeze(), a_0)
    print(h_t)
    z_t = dynamics_mdl(h_t).mean
    x_t = dec(h_t.unsqueeze(0), z_t.unsqueeze(0)).mean
    x_hat_img = cv2.resize(np.moveaxis((255 * torch.clip(x_t, 0, 1)[0]).cpu().detach().numpy().astype("uint8"), 0, -1), display_shape[:2])
    print(x_hat_img)
    cv2.imshow("Imagination", x_hat_img)
    cv2.waitKey(1000)

