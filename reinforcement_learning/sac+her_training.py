import gymnasium as gym
import numpy as np
import torch
import os
from robotenv import HERRobotEnv
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import (
    GoalSelectionStrategy,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList


def make_env():
    """
    Utility to create and wrap a single HERRobotEnv in Monitor.
    This will be used by DummyVecEnv.
    """
    env = HERRobotEnv()
    return Monitor(env)


if __name__ == "__main__":
    train_venv = None
    eval_venv = None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 1) Create vectorized training env with normalization first
        train_venv = DummyVecEnv([make_env])
        train_venv = VecNormalize(
            train_venv,
            norm_obs=True,     # normalize observations
            norm_reward=False, # keep sparse reward untouched
            clip_obs=10.0,
        )

        # 2) Create vectorized evaluation env (no norm updates)
        eval_venv = DummyVecEnv([make_env])
        eval_venv = VecNormalize(
            eval_venv,
            norm_obs=True,
            norm_reward=False,
            training=False,    # do NOT update running stats at eval time
            clip_obs=10.0,
        )

        # Set up callbacks with early stopping
        eval_callback = EvalCallback(
            eval_env=eval_venv,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=1000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=5,
                verbose=1
            )
        )

        # Check if we should initialize from a previous model
        init_model = os.getenv("INIT_MODEL")
        if init_model and os.path.exists(init_model):
            print(f"Initializing from previous model: {init_model}")
            # Load model with the environment
            base_model = SAC.load(init_model, env=train_venv)
            policy_kwargs = base_model.policy_kwargs
            actor_dict = base_model.policy.state_dict()
        else:
            print("Starting fresh training")
            policy_kwargs = dict(
                net_arch=[128, 128],
                activation_fn=torch.nn.ReLU,
            )
            actor_dict = None

        # Build the HER+SAC model
        model = SAC(
            "MultiInputPolicy",      # for Dict obs
            train_venv,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,    # Reduced from 8 to focus more on real experiences
                goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            ),
            learning_rate=3e-4,      # Increased from 1e-5 for faster learning
            buffer_size=100000,      # Increased buffer size
            learning_starts=1000,    # Reduced to start learning earlier in curriculum
            ent_coef="auto",         # Automatic entropy tuning
            batch_size=256,          # Larger batch size
            tau=0.005,              # Slower target network update
            gamma=0.98,             # Slightly reduced discount factor
            train_freq=1,           # Update every step
            gradient_steps=1,       # One gradient step per update
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256, 256],  # Deeper actor network
                    qf=[256, 256, 256]   # Deeper critic network
                ),
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            tensorboard_log="./sac_ppo/",
            device=device,
        )

        # Load weights if we're initializing from a previous model
        if actor_dict is not None:
            model.policy.load_state_dict(actor_dict, strict=False)
            print("Loaded weights from previous model")

        print("Starting HER + SAC training with normalized observations...")
        model.learn(
            total_timesteps=100000,  # Increased training time
            callback=eval_callback,
            log_interval=10,
        )

        # Save final model (best model was already saved by callback)
        model.save("her_sac_robot_final")
        train_venv.save("her_sac_robot_final.stats")
        print("Final model and normalization stats saved!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Always clean up
        if train_venv is not None:
            train_venv.close()
        if eval_venv is not None:
            eval_venv.close()
        print("Environments closed")