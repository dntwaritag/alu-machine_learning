#!/usr/bin/env python3
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """
    Uses the epsilon-greedy policy to determine the next action.
    
    Parameters:
    - Q (numpy.ndarray): The Q-table
    - state (int): The current state
    - epsilon (float): The epsilon value for exploration-exploitation trade-off
    
    Returns:
    - int: The next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])  # Explore: random action
    else:
        return np.argmax(Q[state])  # Exploit: best known action
