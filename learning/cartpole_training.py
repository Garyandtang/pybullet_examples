"""Script demonstrating the use of `gym_pybullet_drones`' Gymnasium interface.

Class HoverAviary is used as a learning env for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning libraries `stable-baselines3`.

"""
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


from envs.cartpole.cartpole import CartPole

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False




if __name__ == "__main__":
    env = gym.make("Cartpole-v1")
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

