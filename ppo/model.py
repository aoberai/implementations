import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PPO(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x))
        value = self.value_head(x)
        return (policy, value)


