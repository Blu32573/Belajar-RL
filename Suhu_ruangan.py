import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import TD3
from stable_baselines3 import SAC
import os
import numpy as np
from gym import spaces
from gym import Env

class Lingkungan_anjayyy(gym.Env) :
    def __init__ (self) :
        super().__init__() 
        self.action_space = spaces.Discrete(2) #saya mau gerak 2 doang, naik/turun
        self.observation_space = spaces.Box(low = 0.0, high = 16.0, shape = (1,), dtype = np.float32)
        self.state = 0.0
        self.max_steps = 50
        self.current_step = 0
        self.target_min = 18.0 
        self.target_max = 25.0

    def reset(self) :
        super().reset(seed = None) 
        self.state = np.array([np.random.uniform(15.0, 35.0)], dtype=np.float32)
        self.current_step = 0
        return self.state
    
    def step(self, action):
        temperature = self.state[0]
        if action == 0:
            temperature -= 2.0
        elif action == 1:
            temperature += 1.0
        temperature = np.clip(temperature, 0.0, 40.0)
        self.state = np.array([temperature], dtype=np.float32)
        reward = 1.0 if self.target_min <= temperature <= self.target_max else -1.0
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Gabungkan terminated dan truncated menjadi done
        return self.state, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, Temperature: {self.state[0]}°C")


"""
episodes = 5
env = Lingkungan_anjayyy()

for episode in range(1, episodes + 1):
    score = 0
    obs, info = env.reset()
    over = False
    while not over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()
        if terminated or truncated:
            over = True
    print('Episode:{} Score:{}'.format(episode, score))
"""

""" 
Hasil  = 
Episode:1 Score:-38.0
Episode:2 Score:-30.0
Episode:3 Score:-34.0
Episode:4 Score:-46.0
Episode:5 Score:-24.0
"""
"""
anjayyy = Lingkungan_anjayyy()
aa = DummyVecEnv([lambda: anjayyy])

model = PPO("MlpPolicy", aa, verbose = 1)
model.learn(total_timesteps = 10000)
model.save("PPO_anjayyy") 
"""

'''
Using cpu device
-----------------------------
| time/              |      |
|    fps             | 1037 |
|    iterations      | 1    |
|    time_elapsed    | 1    |
|    total_timesteps | 2048 |
-----------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 722          |
|    iterations           | 2            |
|    time_elapsed         | 5            |
|    total_timesteps      | 4096         |
| train/                  |              |
|    approx_kl            | 0.0146319885 |
|    clip_fraction        | 0.144        |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.679       |
|    explained_variance   | 0.00199      |
|    learning_rate        | 0.0003       |
|    loss                 | 19.5         |
|    n_updates            | 10           |
|    policy_gradient_loss | -0.0151      |
|    value_loss           | 52.3         |
------------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 630          |
|    iterations           | 3            |
|    time_elapsed         | 9            |
|    total_timesteps      | 6144         |
| train/                  |              |
|    approx_kl            | 0.0072261286 |
|    clip_fraction        | 0.0702       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.647       |
|    explained_variance   | 0.0162       |
|    learning_rate        | 0.0003       |
|    loss                 | 32.3         |
|    n_updates            | 20           |
|    policy_gradient_loss | -0.00854     |
|    value_loss           | 67.2         |
------------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 609         |
|    iterations           | 4           |
|    time_elapsed         | 13          |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.009754886 |
|    clip_fraction        | 0.104       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.637      |
|    explained_variance   | 0.0174      |
|    learning_rate        | 0.0003      |
|    loss                 | 39.6        |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0111     |
|    value_loss           | 73.4        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 593         |
|    iterations           | 5           |
|    time_elapsed         | 17          |
|    total_timesteps      | 10240       |
| train/                  |             |
|    approx_kl            | 0.017223442 |
|    clip_fraction        | 0.136       |
| train/                  |             |
|    approx_kl            | 0.017223442 |
|    clip_fraction        | 0.136       |
|    approx_kl            | 0.017223442 |
|    clip_fraction        | 0.136       |
|    clip_fraction        | 0.136       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.632      |
|    explained_variance   | 0.00587     |
|    entropy_loss         | -0.632      |
|    explained_variance   | 0.00587     |
|    learning_rate        | 0.0003      |
|    explained_variance   | 0.00587     |
|    learning_rate        | 0.0003      |
|    loss                 | 30.6        |
|    learning_rate        | 0.0003      |
|    loss                 | 30.6        |
|    n_updates            | 40          |
|    loss                 | 30.6        |
|    n_updates            | 40          |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0151     |
|    policy_gradient_loss | -0.0151     |
|    value_loss           | 57.4        |
-----------------------------------------'''
"""
model = PPO.load('PPO_anjayyy')

lingkunganku = Lingkungan_anjayyy()

a = 5
for b in range(1, a+1):
    score = 0
    obs = lingkunganku.reset()
    done = False
    while not done :
        action, _ = model.predict(obs)
        obs, reward, done, info = lingkunganku.step(action)
        score += reward
        lingkunganku.render()
    print('Episode:{} Score:{}'.format(b, score))

"""

# Hasil dari  total_timesteps = 10000 adalah
# Episode:1 Score:36.0
# Episode:2 Score:24.0
# Episode:3 Score:2.0
# Episode:4 Score:42.0
# Episode:5 Score:38.0

anjayyy = Lingkungan_anjayyy()
aa = DummyVecEnv([lambda: anjayyy])

model = PPO("MlpPolicy", aa, verbose = 1)
model.learn(total_timesteps = 40000)
model.save("PPO_anjayyy2") 

lingkunganku = Lingkungan_anjayyy()

a = 5
for b in range(1, a+1):
    score = 0
    obs = lingkunganku.reset()
    done = False
    while not done :
        action, _ = model.predict(obs)
        obs, reward, done, info = lingkunganku.step(action)
        score += reward
        lingkunganku.render()
    print('Episode:{} Score:{}'.format(b, score))

"""
# Hasil dari  total_timesteps = 40000 adalah
Using cpu device
-----------------------------
| time/              |      |
|    fps             | 730  |
|    iterations      | 1    |
|    time_elapsed    | 2    |
|    total_timesteps | 2048 |
-----------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 424         |
|    iterations           | 2           |
|    time_elapsed         | 9           |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.014212109 |
|    clip_fraction        | 0.104       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.68       |
|    explained_variance   | 0.00496     |
|    learning_rate        | 0.0003      |
|    loss                 | 12.1        |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0112     |
|    value_loss           | 44.1        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 412         |
|    iterations           | 3           |
|    time_elapsed         | 14          |
|    total_timesteps      | 6144        |
| train/                  |             |
|    approx_kl            | 0.010769013 |
|    clip_fraction        | 0.0532      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.671      |
|    explained_variance   | 0.00716     |
|    learning_rate        | 0.0003      |
|    loss                 | 31.5        |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.00878    |
|    value_loss           | 69.4        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 419         |
|    iterations           | 4           |
|    time_elapsed         | 19          |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.013231128 |
|    clip_fraction        | 0.0686      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.634      |
|    explained_variance   | 0.03        |
|    learning_rate        | 0.0003      |
|    loss                 | 31.4        |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0122     |
|    value_loss           | 62.2        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 418         |
|    iterations           | 5           |
|    time_elapsed         | 24          |
|    total_timesteps      | 10240       |
| train/                  |             |
|    approx_kl            | 0.006535258 |
|    clip_fraction        | 0.0483      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.604      |
|    explained_variance   | 0.0647      |
|    learning_rate        | 0.0003      |
|    loss                 | 20.1        |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0107     |
|    value_loss           | 53.7        |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 431        |
|    iterations           | 6          |
|    time_elapsed         | 28         |
|    total_timesteps      | 12288      |
| train/                  |            |
|    approx_kl            | 0.02160186 |
|    clip_fraction        | 0.111      |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.577     |
|    explained_variance   | 0.0734     |
|    learning_rate        | 0.0003     |
|    loss                 | 17.2       |
|    n_updates            | 50         |
|    policy_gradient_loss | -0.0163    |
|    value_loss           | 42.8       |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 443         |
|    iterations           | 7           |
|    time_elapsed         | 32          |
|    total_timesteps      | 14336       |
| train/                  |             |
|    approx_kl            | 0.015357075 |
|    clip_fraction        | 0.0701      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.571      |
|    explained_variance   | 0.0686      |
|    learning_rate        | 0.0003      |
|    loss                 | 15.1        |
|    n_updates            | 60          |
|    policy_gradient_loss | -0.0115     |
|    value_loss           | 36.9        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 450         |
|    iterations           | 8           |
|    time_elapsed         | 36          |
|    total_timesteps      | 16384       |
| train/                  |             |
|    approx_kl            | 0.010920068 |
|    clip_fraction        | 0.0787      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.584      |
|    explained_variance   | 0.00178     |
|    learning_rate        | 0.0003      |
|    loss                 | 13.4        |
|    n_updates            | 70          |
|    policy_gradient_loss | -0.00546    |
|    value_loss           | 35.9        |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 456        |
|    iterations           | 9          |
|    time_elapsed         | 40         |
|    total_timesteps      | 18432      |
| train/                  |            |
|    approx_kl            | 0.00497303 |
|    clip_fraction        | 0.0899     |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.536     |
|    explained_variance   | 0.00118    |
|    learning_rate        | 0.0003     |
|    loss                 | 20.8       |
|    n_updates            | 80         |
|    policy_gradient_loss | -0.00206   |
|    value_loss           | 45         |
----------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 465          |
|    iterations           | 10           |
|    time_elapsed         | 43           |
|    total_timesteps      | 20480        |
| train/                  |              |
|    approx_kl            | 0.0067984196 |
|    clip_fraction        | 0.0824       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.521       |
|    explained_variance   | 0.000755     |
|    learning_rate        | 0.0003       |
|    loss                 | 30.8         |
|    n_updates            | 90           |
|    policy_gradient_loss | -8.48e-05    |
|    value_loss           | 57.9         |
------------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 471         |
|    iterations           | 11          |
|    time_elapsed         | 47          |
|    total_timesteps      | 22528       |
| train/                  |             |
|    approx_kl            | 0.007184373 |
|    clip_fraction        | 0.0896      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.524      |
|    explained_variance   | -0.000496   |
|    learning_rate        | 0.0003      |
|    loss                 | 26          |
|    n_updates            | 100         |
|    policy_gradient_loss | 0.0028      |
|    value_loss           | 64.9        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 469         |
|    iterations           | 12          |
|    time_elapsed         | 52          |
|    total_timesteps      | 24576       |
| train/                  |             |
|    approx_kl            | 0.004657015 |
|    clip_fraction        | 0.0781      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.532      |
|    explained_variance   | -9.27e-05   |
|    learning_rate        | 0.0003      |
|    loss                 | 37.2        |
|    n_updates            | 110         |
|    policy_gradient_loss | 0.00144     |
|    value_loss           | 73.3        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 475         |
|    iterations           | 13          |
|    time_elapsed         | 55          |
|    total_timesteps      | 26624       |
| train/                  |             |
|    approx_kl            | 0.005332285 |
|    clip_fraction        | 0.0935      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.509      |
|    explained_variance   | 6.02e-05    |
|    learning_rate        | 0.0003      |
|    loss                 | 38.8        |
|    n_updates            | 120         |
|    policy_gradient_loss | 0.00321     |
|    value_loss           | 76          |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 481         |
|    iterations           | 14          |
|    time_elapsed         | 59          |
|    total_timesteps      | 28672       |
| train/                  |             |
|    approx_kl            | 0.010059146 |
|    clip_fraction        | 0.088       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.502      |
|    explained_variance   | 6.42e-05    |
|    learning_rate        | 0.0003      |
|    loss                 | 35.9        |
|    n_updates            | 130         |
|    policy_gradient_loss | 0.00212     |
|    value_loss           | 83.2        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 485         |
|    iterations           | 15          |
|    time_elapsed         | 63          |
|    total_timesteps      | 30720       |
| train/                  |             |
|    approx_kl            | 0.010874853 |
|    clip_fraction        | 0.116       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.511      |
|    explained_variance   | 1.49e-06    |
|    learning_rate        | 0.0003      |
|    loss                 | 55.3        |
|    n_updates            | 140         |
|    policy_gradient_loss | 0.0035      |
|    value_loss           | 82.8        |
-----------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 481          |
|    iterations           | 16           |
|    time_elapsed         | 68           |
|    total_timesteps      | 32768        |
| train/                  |              |
|    approx_kl            | 0.0065174513 |
|    clip_fraction        | 0.0937       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.516       |
|    explained_variance   | -1.6e-05     |
|    learning_rate        | 0.0003       |
|    loss                 | 39           |
|    n_updates            | 150          |
|    policy_gradient_loss | 0.00128      |
|    value_loss           | 85.6         |
------------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 479         |
|    iterations           | 17          |
|    time_elapsed         | 72          |
|    total_timesteps      | 34816       |
| train/                  |             |
|    approx_kl            | 0.006369115 |
|    clip_fraction        | 0.075       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.495      |
|    explained_variance   | 2.38e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 67.9        |
|    n_updates            | 160         |
|    policy_gradient_loss | 0.00184     |
|    value_loss           | 90.7        |
-----------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 481          |
|    iterations           | 18           |
|    time_elapsed         | 76           |
|    total_timesteps      | 36864        |
| train/                  |              |
|    approx_kl            | 0.0021942342 |
|    clip_fraction        | 0.063        |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.495       |
|    explained_variance   | 2.41e-05     |
|    learning_rate        | 0.0003       |
|    loss                 | 44.6         |
|    n_updates            | 170          |
|    policy_gradient_loss | 0.00193      |
|    value_loss           | 89.4         |
------------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 485         |
|    iterations           | 19          |
|    time_elapsed         | 80          |
|    total_timesteps      | 38912       |
| train/                  |             |
|    approx_kl            | 0.010722464 |
|    clip_fraction        | 0.0798      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.496      |
|    explained_variance   | 6.26e-05    |
|    learning_rate        | 0.0003      |
|    loss                 | 39.8        |
|    n_updates            | 180         |
|    policy_gradient_loss | 0.00205     |
|    value_loss           | 91.7        |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 485         |
|    iterations           | 20          |
|    time_elapsed         | 84          |
|    total_timesteps      | 40960       |
| train/                  |             |
|    approx_kl            | 0.008422165 |
|    clip_fraction        | 0.0806      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.487      |
|    explained_variance   | -0.000151   |
|    learning_rate        | 0.0003      |
|    loss                 | 43.8        |
|    n_updates            | 190         |
|    policy_gradient_loss | 0.00272     |
|    value_loss           | 90.8        |
-----------------------------------------"""

"""
Episode:1 Score:46.0
Episode:2 Score:44.0
Episode:3 Score:48.0
Episode:4 Score:46.0
Episode:5 Score:48.0
"""
