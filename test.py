import time
import torch
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import simple_driving


def main():
    for i in range(10):
        # init everything
        env = gym.make('SimpleDriving-v0')
        model = PPO2.load('PPO_SimpleDriving-v0')
        obs = env.reset()
        done = False
        total_reward = 0
        
        # game loop
        while not done:
            time.sleep(1/60)
            env.render()
            action ,_ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        # end game loop
        
        print(f'ep:{i},total_reward{total_reward}')
        env.close()

if __name__ == '__main__':
    main()
