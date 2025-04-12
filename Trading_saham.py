import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from gym import spaces
import pygame
import sys

class Saham_Trulalala(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render_mode=None):
        super(Saham_Trulalala, self).__init__()
        self.render_mode = render_mode  # Inisialisasi render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([10.0, 0.0, 0.0]), high=np.array([100.0, 10000.0, 100000.0]), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 50
        self.saldo = 500.0
        self.saham_saya = 0
        self.harga_saham = 0
        self.price_history = []

        # Inisialisasi pygame jika render_mode adalah "human"
        if self.render_mode == "human":
            self.screen_width = 600
            self.screen_height = 400
            self.screen = None
            self.clock = None

    def reset(self):
        self.step_count = 0
        self.saldo = 500.0
        self.saham_saya = 0 
        self.harga_saham = np.random.uniform(10.0, 100.0)
        self.price_history = [self.harga_saham]
        state = np.array([self.harga_saham, self.saldo, self.saham_saya], dtype=np.float32)
        return state

    def step(self, action):
        self.step_count += 1
        price_charge = np.random.uniform(-0.5, 0.5)
        self.harga_saham = np.clip(self.harga_saham + price_charge, 10.0, 100.0)
        self.price_history.append(self.harga_saham)
        
        total_asset_awal = self.saldo + (self.saham_saya * self.harga_saham)

        if action == 0 and self.saldo >= self.harga_saham:  
            self.saham_saya += 1
            self.saldo -= self.harga_saham

        elif action == 1 and self.saham_saya > 0:  # Jual saham
            self.saham_saya -= 1
            self.saldo += self.harga_saham

        total_asset_akhir = self.saldo + (self.saham_saya * self.harga_saham)

        reward = total_asset_akhir - total_asset_awal

        if action == 2:
            reward -= 0.1

        terminated = self.saldo <= 0
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        state = np.array([self.harga_saham, self.saldo, self.saham_saya], dtype=np.float32)
        return state, reward, done, {}

    def render(self):
        if self.render_mode != "human":
            print(f"Step: {self.step_count}, Harga Saham: {self.harga_saham}, "
                  f"Saham Saya: {self.saham_saya}, Saldo: {self.saldo}")
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Trading Boongan")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        # Tampilkan informasi
        font = pygame.font.Font(None, 36)
        price_text = font.render(f"Price: ${self.harga_saham:.1f}", True, (0, 0, 0))
        balance_text = font.render(f"Balance: ${self.saldo:.1f}", True, (0, 0, 0))
        shares_text = font.render(f"Shares: {self.saham_saya}", True, (0, 0, 0))
        self.screen.blit(price_text, (20, 20))
        self.screen.blit(balance_text, (20, 60))
        self.screen.blit(shares_text, (20, 100))

        # Gambar grafik harga saham
        if len(self.price_history) > 1:
            for i in range(len(self.price_history) - 1):
                x1 = i * (self.screen_width // self.max_steps)
                y1 = self.screen_height - int(self.price_history[i] * 2)
                x2 = (i + 1) * (self.screen_width // self.max_steps)
                y2 = self.screen_height - int(self.price_history[i + 1] * 2)
                pygame.draw.line(self.screen, (0, 0, 255), (x1, y1), (x2, y2), 2)

        pygame.display.flip()
        self.clock.tick(30)

env = Saham_Trulalala(render_mode="human")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)
model.save("trading_model2")

a = 12
for walaweee in range(1, a+1):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Step: {walaweee}, Action: {action}, Reward: {reward}, "
              f"Price: {obs[0]}, Balance: {obs[1]}, Shares: {obs[2]}")