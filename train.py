from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from hexapod_env import *
import shutil
import os

log_dir = "./ppo_hexapod/"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # Hapus folder log lama
    
# Buat environment
# env = HexapodEnv()
env = make_vec_env(HexapodEnv, n_envs=32)

# Inisialisasi model PPO
model = PPO("MlpPolicy", env,
            n_steps=2048,    # Collect 2048 steps before updating
            batch_size=64,   # Train on batches of 64 samples
            n_epochs=10,     # Train each sample 10 times
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log="./ppo_hexapod/")

# Latih model
model.learn(total_timesteps=10000000)

# Simpan model yang sudah dilatih
model.save("hexapod_model")
