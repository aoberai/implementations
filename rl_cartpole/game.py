# https://github.com/aamini/introtodeeplearning/blob/master/lab3/RL.ipynb

# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     # print(env.action_space.sample())
#     env.step(0) # take a random action
#     env.step(env.action_space.sample()) # take a random action
# env.close()
#
import gym
# env = gym.make('CartPole-v1')
env = gym.make('FrozenLake-v0')

# env is created, now we can use it: 
for episode in range(100):
    obs = env.reset()
    counter = 0
    while True:
        counter+=1
        env.render()
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)

        # print(env.action_space)
        #> Discrete(2)
        # print(env.observation_space)
        #> Box(4,)

        # print("Obs: ", nobs, " Reward: ", reward, " Done: ", done, " Info: ", info)
        if done == True:
            print("Episode {} finished after {} timesteps".format(episode + 1, counter+1))
            break

