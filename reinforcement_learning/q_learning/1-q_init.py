#!/usr/bin/env python3
import numpy as np

def q_init(env):
    """
    Initializes the Q-table with zeros.
    
    Parameters:
    - env (gym.Env): The FrozenLakeEnv instance
    
    Returns:
    - numpy.ndarray: Q-table initialized to zeros with shape (state_size, action_size)
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
