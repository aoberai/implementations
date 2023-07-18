from models import Encoder, Decoder

import torch
import torch.optim as optim
import gymnasium as gym
import cv2
import numpy as np

env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()

scene_buffer = []

device = torch.device("cuda")

def get_action(state):
    # exploration policy is random
    return env.action_space.sample()

for _ in range(100):
    # agent policy that uses the observation and info
    action = get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    img = env.render()
    scene_buffer.append(img)

    print(np.shape(img))
    cv2.imshow("", img)
    cv2.waitKey(1)

scene_shape = np.shape(scene_buffer[0])
latent_dim = 6
enc = Encoder(scene_shape, latent_dim).to(device)
dec = Decoder(latent_dim, scene_shape).to(device)
opt_enc = optim.AdamW(enc.parameters(), lr=1e-4, amsgrad=True)
opt_dec = optim.AdamW(dec.parameters(), lr=1e-4, amsgrad=True)

for epoch in range(epochs:=40):
    for i in range(buffer_size:=32, len(scene_buffer), buffer_size):
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        x = torch.tensor(np.array(scene_buffer[i-buffer_size:i]), device=device, dtype=torch.float64)
        print(x[-1])
        z = enc(x)
        x_hat = dec(z)
        print((x - x_hat) ** 2)




env.close()
