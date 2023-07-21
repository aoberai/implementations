from models import Encoder, Decoder

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

env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()
device = torch.device("cuda")

def get_action(state):
    # exploration policy is random
    return env.action_space.sample()

obs_shape = (75, 75, 3)

if sys.argv[1] == "train":
    scene_buffer = []

    for _ in range(20000):
        # agent policy that uses the observation and info
        action = get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        img = env.render()
        scene_buffer.append(cv2.resize(img, obs_shape[:2]))
        cv2.imshow("Win", cv2.resize(scene_buffer[-1], (400, 400)))
        cv2.waitKey(1)

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
            from IPython import display
    plt.ion()

    scene_shape = np.shape(scene_buffer[0])
    # print(scene_shape)
    latent_dim = 10
    enc = Encoder(scene_shape, latent_dim).to(device)
    # print(enc)
    dec = Decoder(latent_dim, scene_shape).to(device)
    opt_enc = optim.AdamW(enc.parameters(), lr=1e-3, amsgrad=True)
    opt_dec = optim.AdamW(dec.parameters(), lr=1e-3, amsgrad=True)

    # Taken from pytorch dqn page
    def plot_durations(eps_returns, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(eps_returns, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Epochs')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    losses = []

    # # make sequences within batches

    # for scene in scene_buffer

    for epoch in range(epochs:=15):
        epoch_losses = 0
        i = 0
        for i in range(buffer_size:=64, len(scene_buffer), buffer_size):
            i += 1
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            x = torch.tensor(np.array(scene_buffer[i-buffer_size:i])/255., device=device, dtype=torch.float).permute(0, 3, 1, 2)
            z = enc(x)
            x_hat = dec(z)
            # print(torch.distributions.Independent(torch.distributions.Normal(x_hat, 1), len(obs_shape)))
            # print(torch.distributions.Independent(torch.distributions.Normal(x_hat, 1), len(obs_shape)).log_prob(x))
            loss = -torch.distributions.Normal(x_hat, 5).log_prob(x).sum()
            # loss = (torch.sum((x - x_hat) ** 2))
            epoch_losses += loss.item()
            loss.backward()
            opt_enc.step()
            opt_dec.step()

        losses.append(epoch_losses/i)
        print("Epoch", epoch, "loss:", epoch_losses)
        plot_durations(losses[1:], show_result=True)
        np.random.shuffle(scene_buffer)

    plot_durations(losses, show_result=True)
    plt.ioff()
    plt.show()

    torch.save(enc, "models/enc.pt")
    torch.save(dec, "models/dec.pt")

elif sys.argv[1] == "inference":
    enc = torch.load("models/enc.pt").to(device)
    dec = torch.load("models/dec.pt").to(device)
    observation, info = env.reset()

    while True:
        action = get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        scene = env.render()
        x = torch.tensor(cv2.resize(scene, obs_shape[:2]), device=device, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2) / 255.
        z = enc(x)
        x_hat = torch.clip(dec(z), 0, 1)
        x_img = cv2.resize(np.moveaxis((255. * x).cpu().numpy().astype("uint8")[0], 0, -1), (400, 400))
        x_hat_img = cv2.resize(np.moveaxis((255 * x_hat).cpu().detach().numpy().astype("uint8")[0], 0, -1), (400, 400))
        # print(np.max(x_hat_img), np.min(x_hat_img))
        # print(torch.max(x_hat), torch.min(x_hat))
        cv2.imshow("Original | AE'd", cv2.hconcat([x_img, x_hat_img]))
        cv2.waitKey(1)
