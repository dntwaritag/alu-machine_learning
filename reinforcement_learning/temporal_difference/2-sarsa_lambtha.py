#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Selects action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Perform SARSA(λ) algorithm with epsilon-greedy exploration.

    Parameters:
    - env: environment instance
    - Q (numpy.ndarray): array of shape (s, a) with state-action values
    - lambtha (float): eligibility trace factor
    - episodes (int): number of episodes to train over
    - max_steps (int): max steps per episode
    - alpha (float): learning rate
    - gamma (float): discount rate
    - epsilon (float): initial epsilon value for epsilon-greedy
    - min_epsilon (float): minimum epsilon value
    - epsilon_decay (float): rate of epsilon decay per episode

    Returns:
    - Q (numpy.ndarray): updated Q table
    """

    initial_epsilon = epsilon

    for episode in range(episodes):
        # Reset and choose first action
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        # Init. eligibility traces to zero, for all states
        eligibility_traces = np.zeros_like(Q)

        for steps in range(max_steps):
            # Take the action in the environment
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action based on epsilon-greedy policy
            new_action = epsilon_greedy(Q, new_state, epsilon)

            # TD Error (δ): reward + gamma * V(next_state) - V(state)
            delta = (reward + (gamma * Q[new_state, new_action]) -
                     Q[state, action])

            # Update eligibility traces, apply lambtha decay
            eligibility_traces[state, action] += 1
            eligibility_traces *= lambtha * gamma

            # Update the Q-table
            Q += alpha * delta * eligibility_traces

            # Update to the next state & action
            state = new_state
            action = new_action

            if terminated or truncated:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
