import gymnasium as gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
#from keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy for compatibility
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Wrapping the environment for compatibility with keras-rl2
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

tf.get_logger().setLevel('ERROR')
# Create environment with necessary wrappers
def create_environment():
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = ResizeObservation(env, (84, 84))         # Resize frames to 84x84 for input
    env = GrayScaleObservation(env)                # Convert frames to grayscale
    env = FrameStack(env, 4)                       # Stack 4 frames
    return env

# Build the DQN model
def build_model(action_size, input_shape=(84, 84, 4)):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    return model

# Train the agent
def train():
    env = create_environment()
    nb_actions = env.action_space.n  # Number of actions in Breakout

    # Build the DQN model
    model = build_model(nb_actions)

    # Set up memory and policy
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    # Configure the DQN agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train the DQN agent
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)

    # Save the trained policy network
    dqn.save_weights('policy.h5', overwrite=True)

    # Close the environment
    env.close()

if __name__ == "__main__":
    train()