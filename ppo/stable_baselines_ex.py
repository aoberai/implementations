# playing around with stable baselines premade PPO implementation

import gymnasium as gym
import cv2

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=4 * 25000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    cv2.imshow("win", vec_env.render())
    cv2.waitKey(1)
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
