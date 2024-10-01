import gymnasium as gym
import numpy as np
import torch
from robotenv import RobotEnv
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

class RewardThresholdCallback(BaseCallback):
    def __init__(self, reward_threshold, check_freq=1000, verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.total_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals['rewards']) > 0:
            self.total_rewards.append(self.locals['rewards'][-1])  # Append last reward

        if self.n_calls % self.check_freq == 0:
            average_reward = np.mean(self.total_rewards[-self.check_freq:])  # Last `check_freq` rewards
            if average_reward >= self.reward_threshold:
                print(f"Reached reward threshold: {average_reward}")
                return False  # Stop training
        return True

def make_env():
    def _init():
        env = RobotEnv() 
        check_env(env)
        return Monitor(env)
    return _init

if __name__ == '__main__':  # <-- This is the key change
    # Number of parallel environments (robots)
    num_envs = 40  # You can change this to the desired number of parallel robots

    # Create the parallel environment
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Checks if GPU is available
    # Create the SAC model
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_robot_tensorboard/")

    # Set your desired reward threshold
    reward_threshold = 100  # Adjust this to your preference
    reward_callback = RewardThresholdCallback(reward_threshold=reward_threshold)

    # Train the model
    model.learn(total_timesteps=1000000000, callback=[eval_callback, reward_callback])

    # Save the model
    model.save("sac_robot")

    # Test the trained model
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if np.any(done):  # Check if any environment is done
            obs = env.reset()  # Reset all environments

    env.close()
