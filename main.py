
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


class Policy_Network(nn.Module):
    def __init__(self):
        super().__init__()


class Reinforce_Agent:
    def __init__(self):
        self.GAMMA = 0.99
        self.LEARNING_RATE = 0.001
        self.ALPHA = 1e-3
        self.epsilon = 1e-6
    
    SEED = 0
    MEMORY_SIZE = 100_000     # size of memory buffer
    GAMMA = 0.995             # discount factor
    ALPHA = 1e-3              # learning rate  
    NUM_STEPS_FOR_UPDATE = 7  # perform a learning update every C time steps
    MINIBATCH_SIZE = 64  # Mini-batch size.
    TAU = 1e-3  # Soft update parameter.
    E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
    E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.


if __name__ == "__main__":
    print("Hello World!")