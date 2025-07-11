#!/usr/bin/env python3
import numpy as np

def play(env, Q, max_steps=100):
    """
    Have the trained agent play an episode in the environment.

    Parameters:
    env (gym.Env): The FrozenLake environment.
    Q (numpy.ndarray): The Q-table containing action values.
    max_steps (int): The maximum number of steps in the episode.

    Returns:
    float: The total reward for the episode.
    """
    state = env.reset()  # Reset the environment to start a new episode
    total_reward = 0  # Initialize the total reward

    for _ in range(max_steps):
        # Render the environment state
        env.render()

        # Select the action with the highest Q-value for the current state (exploit)
        action = np.argmax(Q[state])

        # Take the action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)

        # Update the total reward
        total_reward += reward

        # If the agent reaches the goal or falls in a hole, end the episode
        if done:
            break

        # Update the state
        state = next_state

    # Render the final state
    env.render()

    return total_reward
