import numpy as np
import random
import gym

# Import environment
from hexapod_env import HexapodEnv

def discretize_state(state, bins):
    """Membagi state menjadi indeks diskret untuk digunakan dalam tabel Q."""
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))

# Inisialisasi environment
env = HexapodEnv()

# Parameter Q-learning
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

action_space_size = 10  # Jumlah diskret untuk aksi
state_bins = [np.linspace(-1, 1, 10) for _ in range(env.observation_space.shape[0])]

table_shape = tuple(len(b) + 1 for b in state_bins) + (action_space_size,)
Q_table = np.random.uniform(low=-1, high=1, size=table_shape)

for episode in range(episodes):
    print(f"episode: {episode}")
    state = env.reset()
    state = discretize_state(state, state_bins)
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action_idx = np.random.randint(action_space_size)
        else:
            action_idx = np.argmax(Q_table[state])
        
        action = np.array([-0.1 + (0.2 / action_space_size) * action_idx, 0])
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state, state_bins)

        # Update Q-table
        best_next_action = np.argmax(Q_table[next_state])
        Q_table[state + (action_idx,)] += alpha * (reward + gamma * Q_table[next_state + (best_next_action,)] - Q_table[state + (action_idx,)])
        
        state = next_state
        total_reward += reward

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
