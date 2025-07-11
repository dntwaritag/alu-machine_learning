#!/usr/bin/env python3
import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from OpenAI’s gym.
    
    Parameters:
    - desc (list of lists, optional): Custom description of the map
    - map_name (str, optional): Pre-made map to load
    - is_slippery (bool): Determines if the ice is slippery
    
    Returns:
    - gym.Env: The FrozenLake environment instance
    """
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)
    
    return gym.make("FrozenLake-v1", desc=desc, map_name=map_name, is_slippery=is_slippery)
