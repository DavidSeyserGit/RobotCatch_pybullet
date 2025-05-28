# ppo_finetune_simple.py

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from robotenv import RobotEnv

def make_env():
    """Constructs one Monitor-wrapped RobotEnv."""
    def _init():
        env = RobotEnv()
        return Monitor(env)
    return _init

def main():
    # 1) build a single-env vector wrapper for simplicity
    env = DummyVecEnv([make_env()])

    # 2) load your SAC+HER model (saved via sac.save("sac_her_robot.zip"))
    sac = SAC.load("sac_her_robot.zip", env=env)

    # 3) extract its policy architecture and actor weights
    policy_kwargs = sac.policy_kwargs
    actor_dict = sac.policy.state_dict()

    # 4) build a new PPO with the same MlpPolicy & policy_kwargs
    ppo = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_simple_tb/",
    )

    # 5) copy over the SAC actor weights into PPO’s policy network
    ppo.policy.load_state_dict(actor_dict, strict=False)

    # 6) train for ~100 episodes = 100 * 500 steps ≈ 50 000 timesteps
    total_timesteps = 100 * 500
    ppo.learn(total_timesteps=total_timesteps)

    # 7) save your fine-tuned PPO
    ppo.save("ppo_robot_finetuned_simple.zip")

    env.close()

if __name__ == "__main__":
    main()