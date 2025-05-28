import gymnasium as gym
import numpy as np
import torch
from robotenv import HERRobotEnv
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import (
    GoalSelectionStrategy,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class SuccessRateCallback(BaseCallback):
    """
    Callback for tracking success rate in HER environments
    """
    def __init__(self, success_threshold=0.8, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_threshold = success_threshold
        self.successes = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # infos is a list (one entry per env in the VecEnv)
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "episode" in infos[0]:
            self.episode_count += 1
            success = infos[0].get("is_success", False)
            self.successes.append(1 if success else 0)

        if (
            self.episode_count > 0
            and self.episode_count % self.check_freq == 0
        ):
            recent = self.successes[-self.check_freq :]
            rate = np.mean(recent) if recent else 0.0
            if self.verbose > 0:
                print(
                    f"[Callback] Episodes: {self.episode_count}  "
                    f"Success rate: {rate:.3f}"
                )
            if rate >= self.success_threshold:
                print(f"[Callback] Reached success threshold {rate:.3f}")
                return False  # stop training
        return True


def make_env():
    """
    Utility to create and wrap a single HERRobotEnv in Monitor.
    This will be used by DummyVecEnv.
    """
    env = HERRobotEnv()
    return Monitor(env)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Create vectorized training env with normalization
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

    # 3) Set up EvalCallback to run every 5000 steps on the normalized eval env
    eval_callback = EvalCallback(
        eval_venv,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # 4) Our success‐rate callback (works on the training VecEnv)
    success_callback = SuccessRateCallback(
        success_threshold=0.8, check_freq=100, verbose=1
    )

    # 5) Build the HER+SAC model
    model = SAC(
        "MultiInputPolicy",      # for Dict obs
        train_venv,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        ),
        learning_rate=1e-5,
        buffer_size=1000000,
        learning_starts=1000,
        ent_coef=0.01,
        batch_size=64,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        tensorboard_log="./her_sac_robot_tensorboard/",
        device=device,
    )

    print("Starting HER + SAC training with normalized observations...")
    model.learn(
        total_timesteps=int(1e5),  # ramp this up to 1e6+ for better results
        callback=[eval_callback, success_callback],
        log_interval=10,
    )

    model.save("her_sac_robot")
    print("Model saved!")

    # 6) Final test (we still want to test on the same normalized train_venv)
    successes = 0
    episodes = 100
    for ep in range(episodes):
        obs = train_venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # VecEnv.step() returns 4 values: obs, reward, done, infos
            obs, reward, done, infos = train_venv.step(action)
            # done is already combined (terminated OR truncated)
        
        if infos[0].get("is_success", False):
            successes += 1

    rate = successes / episodes
    print(f"Final test success rate: {rate:.3f} ({successes}/{episodes})")
    # 7) Clean up
    train_venv.close()
    eval_venv.close()