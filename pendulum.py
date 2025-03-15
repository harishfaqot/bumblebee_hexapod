import gym
import numpy as np
import random

# Membuat lingkungan inverted pendulum
# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')

# Menentukan parameter Q-learning
alpha = 0.3  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Epsilon-greedy factor
num_episodes = 1000  # Jumlah episode

# Inisialisasi Q-table
state_space_size = (24, 24, 24, 24)  # Pembagian state space menjadi grid (sudah disesuaikan dengan CartPole)
q_table = np.random.uniform(low=-1, high=1, size=state_space_size + (env.action_space.n,))

def discretize_state(state):
    """
    Mendiskritisasi state menjadi indeks dalam Q-table.
    """
    bins = [np.linspace(-2.4, 2.4, num=24),
            np.linspace(-3.0, 3.0, num=24),
            np.linspace(-0.5, 0.5, num=24),
            np.linspace(-3.0, 3.0, num=24)]
    
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_indices)

def choose_action(state):
    """
    Memilih aksi menggunakan epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Pilih aksi acak
    else:
        state_indices = discretize_state(state)
        return np.argmax(q_table[state_indices])  # Pilih aksi berdasarkan Q-table

def update_q_table(state, action, reward, next_state):
    """
    Memperbarui Q-table berdasarkan pengalaman.
    """
    state_indices = discretize_state(state)
    next_state_indices = discretize_state(next_state)
    
    best_next_action = np.argmax(q_table[next_state_indices])
    q_table[state_indices + (action,)] = q_table[state_indices + (action,)] + alpha * (reward + gamma * q_table[next_state_indices + (best_next_action,)] - q_table[state_indices + (action,)])

# Q-learning loop
for episode in range(num_episodes):
    state, _ = env.reset()  # Mengambil state awal dan mengabaikan informasi tambahan
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        update_q_table(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        
    print(f"Episode {episode}: Total Reward = {total_reward}")

# Menyelesaikan simulasi untuk episode terakhir
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
done = False
while not done:
    action = choose_action(state)
    state, _, done, _, _ = env.step(action)
    env.render()

env.close()
