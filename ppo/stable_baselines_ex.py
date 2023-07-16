# playing around with stable baselines premade PPO implementation

import gymnasium as gym
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from custom_env.gym_env import ChopperScape

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = ChopperScape()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20 * 25000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    cv2.imshow("win", vec_env.render())
    cv2.waitKey(1)
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
