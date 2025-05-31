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
                n_sampled_goal=8,
                goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            ),
            learning_rate=1e-5,
            buffer_size=50000,
            learning_starts=1000,
            ent_coef=0.01,
            batch_size=64,
            tau=0.01,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
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
            total_timesteps=2000,
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