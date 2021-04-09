import time
import torch
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import simple_driving

def main():
    env = gym.make('SimpleDriving-v0')
    model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./tensorboard/").learn(total_timesteps=100000)
    model.save('PPO_SimpleDriving-v0')
    del model
    env.close()

if __name__ == '__main__':
    main()
