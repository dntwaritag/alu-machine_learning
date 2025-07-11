#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform the TD(λ) algorithm to update the value function V.

    Parameters:
    - env: environment instance
    - V (numpy.ndarray): array of shape (s,) with state values
    - policy: function to select actions
    - lambtha (float): eligibility trace factor
    - episodes (int): number of episodes to train over
    - max_steps (int): max steps per episode
    - alpha (float): learning rate
    - gamma (float): discount rate

    Returns:
    - V (numpy.ndarray): updated value estimate
    """

    for episode in range(episodes):
        state = env.reset()[0]

        # Initialize eligibility to 0
        trace = np.zeros_like(V)

        # Generate an episode
        for t in range(max_steps):
            action = policy(state)  # Get action from policy

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Calculate TD error (delta)
            delta = reward + gamma * V[next_state] - V[state]
            trace[state] += 1

            # Update the value for all states
            V += alpha * delta * trace

            # Update the estimates for all states
            trace *= gamma * lambtha

            state = next_state  # Update state

            if done:
                break  # End episode if 'done' signal is received

    return V
