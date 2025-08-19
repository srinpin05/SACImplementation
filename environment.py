import gymnasium as gym
import tensorflow as tf
import panda_gym
import time 
env = gym.make("PandaPushDense-v3", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action for demonstratio
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.20)
    if terminated or truncated:
        observation, info = env.reset()
        print("Resetting environment")
env.close()
print("Finished running the environment")


