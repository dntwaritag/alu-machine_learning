#!/usr/bin/env python3
import numpy as np

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform Q-learning on the FrozenLake environment.

    Parameters:
    env (gym.Env): The environment to train in.
    Q (numpy.ndarray): The Q-table.
    episodes (int): The number of training episodes.
    max_steps (int): The maximum number of steps per episode.
    alpha (float): The learning rate.
    gamma (float): The discount factor.
    epsilon (float): The initial exploration rate.
    min_epsilon (float): The minimum value for epsilon.
    epsilon_decay (float): The decay rate for epsilon.

    Returns:
    Q (numpy.ndarray): The updated Q-table.
    total_rewards (list): A list of rewards per episode.
    """
    total_rewards = []  # This will store the rewards for each episode

    for episode in range(episodes):
        state = env.reset()  # Reset the environment and get the starting state
        total_reward = 0  # Initialize total reward for this episode
        
        for step in range(max_steps):
            # Choose action based on epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(Q[state])  # Best action based on the Q-table
            
            # Take the action, observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            
            # If the agent falls into a hole (done is True), set the reward to -1
            if done and reward == 0:
                reward = -1
            
            # Update the Q-value using the Q-learning formula
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            total_reward += reward
            state = next_state  # Move to the next state
            
            if done:
                break
        
        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        total_rewards.append(total_reward)  # Store the total reward for this episode

    return Q, total_rewards
