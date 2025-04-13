import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from gym import spaces
import pygame
import sys

class TrafficMejikuhi(gym.Env) :
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(TrafficMejikuhi, self).__init__()
        self.action_space = spaces.Discrete(3) # 0: Merah, 1: Kuning, 2: Hijau
        self.observation_space = spaces.Box(low=np.array([0.0]), high=np.array([50.0]), dtype=np.float32)
        self.current_count = 0.0
        self.max_steps = 50.0
        self.jumlah_mobil = 0
        self.state = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_count = 0
        self.state = np.array([np.random.uniform(0.0, 20.0)], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        self.current_count += 1
        mobil_datang = np.random.uniform(1.0, 10.0)
        Banyak_mobil = np.clip(self.state[0] + mobil_datang, 0.0, 50.0)

        if action == 0:  
            Banyak_mobil += mobil_datang
        elif action == 2:  
            Banyak_mobil -= 5.0

        Banyak_mobil += 0.2
        # Hitung reward
        reward = 0.0
        if Banyak_mobil >= 40.0:
            reward -= 2.0
        else:
            reward += 1.0

        # Perbarui state
        self.state = np.array([Banyak_mobil], dtype=np.float32)

        terminated = Banyak_mobil > 50.0  
        done = self.current_count >= self.max_steps  
        info = {}

        return self.state, reward, terminated, done, info

    def render(self):
        print(f"Step: {self.current_count}, Number of Cars: {self.state}, Points: {reward}")

env = TrafficMejikuhi()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=40000)

episode = 10
for a in range(1, episode + 1):
    obs, _ = env.reset()
    done = False
    terminated = False
    total_reward = 0
    while not (done or terminated):  # Periksa kedua kondisi
        action, _ = model.predict(obs)
        obs, reward, terminated, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"Episode {a}: Total Reward: {total_reward}")
    
        


