import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot

env = gym.make('turtlebot-v0')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1)
