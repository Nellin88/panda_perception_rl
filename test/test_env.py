import os
import sys
import time

# Allow running this file directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import panda_mujoco_gym

if __name__ == "__main__":
    env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.2)

    env.close()
