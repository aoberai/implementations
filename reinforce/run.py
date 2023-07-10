from model import REINFORCE
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
from torch.distributions import Categorical

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
        from IPython import display
plt.ion()

"""
CartPole observation space: [cart position (-4.8, 4.8), cart velocity (-inf, inf), pole angle (-24, 24 deg), pole angular velocity]
Rewards: +1 reward for every step, including termination. Max of 475
Actions: 0: push cart to left; 1: push cart to the right
"""
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
render = True

device = torch.device("cuda")
policy_model = REINFORCE(env.observation_space.shape[0], env.action_space.n).to(device)
opt = optim.AdamW(policy_model.parameters(), lr=1e-4, amsgrad=True)

# Taken from pytorch dqn page
def plot_durations(eps_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(eps_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
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

def returns(rews, discount_fac=0.95):
    g = 0
    for i in range(len(rews)):
        g += (discount_fac ** i) * rews[i]

    return g

if __name__ == "__main__":
    eps_durations = []
    for eps in range(episodes:=1500):
        obs, __ = env.reset()
        actions = []
        done = False
        steps = 0

        eps_obs, eps_action_prob, eps_rews = [], [], []
        while not done:
            policy_distr = policy_model(torch.tensor(obs, device=device))
            m = Categorical(policy_distr)
            action = int(m.sample())
            # print(action)
            action_prob = policy_distr[action]
            # print(action, action_prob)

            eps_obs.append(obs)
            eps_action_prob.append(action_prob)

            obs, rew, term, trunc, __ = env.step(action)

            eps_rews.append(rew)

            done = term or trunc
            actions.append(action)
            # print(obs, term, trunc, np.mean(actions))

            steps += 1
        
        eps_durations.append(steps)
        plot_durations(eps_durations)
   
        rewards_to_go = []
        for i in range(len(eps_rews)):
            rewards_to_go.append(returns(eps_rews[i:]))
        
        rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (np.std(rewards_to_go) + 1e-08)
        print(rewards_to_go)

        losses = []
        for obs, prob, ret in zip(eps_obs, eps_action_prob, rewards_to_go):
            losses.append(-ret * torch.log(prob))
       
        opt.zero_grad()
        loss = torch.stack(losses).sum()
        loss.backward()
        opt.step()

    print("Complete")
    torch.save(policy_model, "reinforce.pt")
    plot_durations(eps_durations, show_result=True)
    plt.ioff()
    plt.show()


