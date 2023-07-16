# playing around with stable baselines premade PPO implementation

import gymnasium as gym
import cv2
import os
import random
import json
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# from custom_env.gym_env import ChopperScape
from custom_env.entropy_env import StateFilterEntropy

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
# env = ChopperScape()
prefix_path = "/home/aoberai/programming/vtol-surveyor/scripts/alphasurveyor/bitmap_info/"
bitmaps = [prefix_path + file for file in os.listdir(prefix_path) if ".json" in file]

bitmap, bitmap_info, bitmap_path, bitmap_img = None, None, None, None

while True:
    try:
        bitmap = random.choice(bitmaps)
        bitmap_info = json.load(open(bitmap))
        bitmap_path = bitmap_info["path"]
        bitmap_img = cv2.imread(bitmap_path)
        break
    except Exception as e:
        print(e)
        pass

cv2.imshow("Augmented Bitboard", bitmap_img)
cv2.waitKey(1000)


env = StateFilterEntropy(20, bitmap_img)
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)

for i in range(10):
    # model.learn(total_timesteps=25000)
    model.learn(total_timesteps=2048*2)

    vec_env = copy.copy(model.get_env())
    vec_env.set_render(mode="human")
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
