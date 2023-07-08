# took boiler plate stuff from here and used for debugging: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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
from model import DQN

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
        from IPython import display
plt.ion()

"""
CartPole observation space: [cart position (-4.8, 4.8), cart velocity (-inf, inf), pole angle (-24, 24 deg), pole angular velocity]
Rewards: +1 reward for every step, including termination. Max of 475
Actions: 0: push cart to left; 1: push cart to the right
"""
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
render = True
reward_discount = 0.98
epsilon = 1
batch_size = 128
buffer_size = 10000

class DQN(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x

device = torch.device("cuda")
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
    policy_model.train()
    batch_states = torch.as_tensor(np.array([elem[0] for elem in batch]), device=device)
    batch_rewards = np.array([elem[3] for elem in batch])
    batch_nx_states = torch.as_tensor(np.array([elem[2] for elem in batch if elem[2] is not None]), device=device)
    batch_fin_states_mask = torch.as_tensor(np.array([(elem[2] is not None) for elem in batch]), device=device)
    batch_action = torch.as_tensor([elem[1] for elem in batch], device=device).unsqueeze(1)
    
    state_action_values = policy_model(batch_states).gather(1, batch_action)
    target_nx_val = torch.zeros(len(batch_rewards), device=device)

    with torch.no_grad():
        target_nx_val[batch_fin_states_mask] = target_model(batch_nx_states).max(1)[0]
    
    expected_action_values = torch.as_tensor(batch_rewards, device=device) + reward_discount * torch.unsqueeze(target_nx_val, axis=1)

    """
    print(np.shape(state_action_values))
    print(np.shape(expected_action_values))
    print(np.shape(reward_discount * torch.unsqueeze(torch.max(target_model(batch_nx_states), axis=1).values, axis=1)))
    print(np.shape(torch.as_tensor(batch_rewards, device=device)))
    """
    """
    print(state_action_values)
    print(len(state_action_values))
    print(expected_action_values)
    print(len(expected_action_values))
    """

    huber_criterion = nn.SmoothL1Loss()
    huber_loss = huber_criterion(state_action_values, expected_action_values) # can't apply chain rule rn because tensors not preserved from model output

    opt.zero_grad()
    huber_loss.backward()
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    opt.step()

    return huber_loss

def get_action(state, steps_done, model):
    # Epsilon greedy policy for selecting action through episodes
    exp_prob = (exp_end:=0.05) + ((exp_init:=0.9) - exp_end) * math.exp(-1 * steps_done/(decay_rate:=1000))
    # print(exp_prob)
    if random.random() < exp_prob:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # exploration
        return env.action_space.sample() # exploration
    else:
        with torch.no_grad():
            action = torch.argmax(model(torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=device))[0]) # exploitation
            # action = torch.argmax(model(torch.tensor(state, dtype=torch.float32, device=device))[0]) # exploitation
            return action

episode_durations = []

# Taken from link above
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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

episode_count = 0
step_count = 0
last_step_count = 0
while episode_count < 600:
    state, info = env.reset()
    # state = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=device) # .unsqueeze(0) instead of np.expand_dims
    print("Buffer Size:", len(buffer))
    term = False

    actions = []
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
        buffer.add(state, action.item(), next_state, [rew])
        actions.append(action.item())

        state = next_state

        if len(buffer) > batch_size:
            train(buffer.sample(batch_size))

        # Update target network weights

        target_model_state_dict = target_model.state_dict()
        policy_model_state_dict = policy_model.state_dict()

        # Soft update of target net
        for key in policy_model_state_dict:
            target_model_state_dict[key] = policy_model_state_dict[key]*(TAU:=0.005) + target_model_state_dict[key]*(1-TAU)
        target_model.load_state_dict(target_model_state_dict)

        if term:
            episode_count += 1
            print("epsde time:", step_count - last_step_count)
            episode_durations.append(step_count - last_step_count)
            break

        # print(obs, rew, term, trunc)
        """
        if render and episode_count % 25 == 0:
            cv2.imshow("", cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(50)
        """
        step_count += 1

    last_step_count = step_count
    print("Episode:", episode_count)
    print("Average action:", np.mean(actions))
    plot_durations()

print("Complete")
torch.save(target_model, "dqn.pt")
plot_durations(show_result=True)
plt.ioff()
plt.show()
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
