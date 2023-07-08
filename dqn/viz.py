import gymnasium as gym
import torch
from model import DQN

device = torch.device('cuda')
policy_model = torch.load("dqn.pt", map_location=device)
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    # agent policy that uses the observation and info
    # action = env.action_space.sample()
    action = torch.argmax(policy_model(torch.as_tensor(observation, device=device).unsqueeze(0))).item()
    print(_, action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
