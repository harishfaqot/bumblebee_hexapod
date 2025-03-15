from stable_baselines3 import PPO
from hexapod_env import *

# Buat environment
env = HexapodEnv()

# Load model yang sudah dilatih
model = PPO.load("hexapod_model")

# Coba model untuk mengendalikan robot
obs = env.reset()
for _ in range(100000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
