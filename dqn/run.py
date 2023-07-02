# took boiler plate stuff from here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import torch
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
CartPole observation space: [cart position (-4.8, 4.8), cart velocity (-inf, inf), pole angle (-24, 24 deg), pole angular velocity]
Rewards: +1 reward for every step, including termination. Max of 475
Actions: 0: push cart to left; 1: push cart to the right
"""
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
render = True
reward_discount = 0.98
epsilon = 1
batch_size = 32
buffer_size = 10000

class DQN(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.head = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model, target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device), DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_model.load_state_dict(policy_model.state_dict())
opt = optim.AdamW(policy_model.parameters(), lr=1e-4, amsgrad=True)

class ReplayBuffer:

    def __init__(self, capacity=1000):
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


def train(batch):
    policy_model.eval()
    batch_states = torch.as_tensor(np.array([elem[0] for elem in batch]), device=device)
    batch_rewards = np.array([elem[3] for elem in batch])
    batch_nx_states = torch.as_tensor(np.array([elem[2] for elem in batch]), device=device)
    batch_action = [elem[1] for elem in batch]
    action_vals = policy_model(batch_states).tolist()
    # print(action_vals)
    # print(batch_action)
    # print(np.array([action_vals[i][batch_action[i]] for i in range(len(batch_action))]))
    # print(policy_model(batch_nx_states).tolist())
    state_action_values = torch.as_tensor(np.array([action_vals[i][batch_action[i]] for i in range(len(batch_action))]), device=device)
    expected_action_values = torch.as_tensor(batch_rewards, device=device) + reward_discount * torch.max(target_model(batch_nx_states), axis=1).values # Make sure batch rewards is fixed through differentiation
    """
    print(state_action_values)
    print(len(state_action_values))
    print(expected_action_values)
    print(len(expected_action_values))
    """

    huber_criterion = nn.SmoothL1Loss()
    huber_loss = huber_criterion(state_action_values, expected_action_values) # can't apply chain rule rn because tensors not preserved from model output

    policy_model.train()
    opt.zero_grad()
    huber_loss.backward()
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    opt.step()
    policy_model.eval()

    return huber_loss

def get_action(state, steps_done, model):
    # Epsilon greedy policy for selecting action through episodes
    exp_prob = (exp_end:=0.05) + ((exp_init:=0.9) - exp_end) * math.exp(-1 * steps_done/(decay_rate:=1000))
    # print(exp_prob)
    if random.random() < exp_prob:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # exploration
        return env.action_space.sample() # exploration
    else:
        model.eval()
        action = torch.argmax(model(torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=device))[0]) # exploitation
        # action = torch.argmax(model(torch.tensor(state, dtype=torch.float32, device=device))[0]) # exploitation
        model.train()
        return action

episode_count = 0
step_count = 0
last_step_count = 0
while True:
    state, info = env.reset()
    # state = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=device) # .unsqueeze(0) instead of np.expand_dims
    term = False
    while not term:
        action = get_action(state, step_count, policy_model)
        obs, rew, term, trunc, _ = env.step(action.item())
        term = term or trunc
        if term:
            next_state = None
        else:
            # next_state = torch.tensor(np.expand_dims(obs,0), dtype=torch.float32, device=device)
            next_state = obs

            # rew = torch.tensor([rew], device=device)
            # obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
            buffer.add(state, action.item(), next_state, rew)
            # print(buffer.last())

        state = next_state

        if len(buffer) > batch_size:
            train(buffer.sample(batch_size))

        # Update target network weights

        target_model_state_dict = target_model.state_dict()
        policy_model_state_dict = policy_model.state_dict()

        # Soft update of target net
        for key in policy_model_state_dict:
            target_model_state_dict[key] = policy_model_state_dict[key]*(TAU:=0.005) + target_model_state_dict[key]*(1-TAU)
            # print(key)
        target_model.load_state_dict(target_model_state_dict)

        if term:
            episode_count += 1
            print("epsde time:", step_count - last_step_count)
            # plot
            break

        # print(obs, rew, term, trunc)
        # if render:
        #     env.render()
        step_count += 1

    last_step_count = step_count
    print("Episode:", episode_count)
    time.sleep(1)

    if episode_count > 500: # temporary
        break

"""
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, terminated, truncated, info)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()

env.close()
"""
