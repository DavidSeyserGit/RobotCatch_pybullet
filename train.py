import gymnasium as gym
import numpy as np
import torch
from robotenv import HERRobotEnv
from stable_baselines3 import SAC, TD3
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class SuccessRateCallback(BaseCallback):
    """
    Callback for tracking success rate in HER environments
    """
    def __init__(self, success_threshold=0.8, check_freq=100, verbose=1):
        super(SuccessRateCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.success_threshold = success_threshold
        self.successes = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check for episode endings and track success
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]  # Single environment
            if 'episode' in info:  # Episode ended
                self.episode_count += 1
                # Check if the episode was successful
                success = info.get('is_success', False)
                self.successes.append(1 if success else 0)

        if self.episode_count > 0 and self.episode_count % self.check_freq == 0:
            recent_successes = self.successes[-self.check_freq:]
            success_rate = np.mean(recent_successes) if recent_successes else 0
            
            if self.verbose > 0:
                print(f"Episodes: {self.episode_count}, Success rate: {success_rate:.3f}")
            
            if success_rate >= self.success_threshold:
                print(f"Reached success threshold: {success_rate:.3f}")
                return False  # Stop training
        
        return True


if __name__ == '__main__':
    # Single environment for HER
    env = HERRobotEnv()
    check_env(env)
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = HERRobotEnv()
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        log_path='./logs/', 
        eval_freq=5000,
        deterministic=True, 
        render=False,
        n_eval_episodes=10
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # SAC with HER - removed online_sampling parameter
    model = SAC(
        "MultiInputPolicy",  # Required for Dict observation spaces
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # Number of HER goals to sample per transition
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            # online_sampling=True,  # Removed this line
        ),
        learning_rate=1e-3,
        buffer_size=1000000,  # Large buffer for HER
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./her_sac_robot_tensorboard/",
        device=device
    )

    # Track success rate instead of reward threshold
    success_callback = SuccessRateCallback(
        success_threshold=0.8,  # Stop when 80% success rate
        check_freq=100,         # Check every 100 episodes
        verbose=1
    )

    print("Starting HER training...")
    # Train the model
    model.learn(
        total_timesteps=int(1e6),  # HER typically needs more timesteps
        callback=[eval_callback, success_callback],
        log_interval=10
    )

    # Save the model
    model.save("her_sac_robot")
    print("Model saved!")

    # Test the trained model
    print("Testing trained model...")
    obs, _ = env.reset()
    successes = 0
    total_episodes = 0
    
    for i in range(100):  # Test for 100 episodes
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            total_episodes += 1
            if info.get('is_success', False):
                successes += 1
                print(f"Episode {total_episodes}: SUCCESS")
            else:
                print(f"Episode {total_episodes}: Failed")
            
            obs, _ = env.reset()
    
    success_rate = successes / total_episodes if total_episodes > 0 else 0
    print(f"Final test success rate: {success_rate:.3f} ({successes}/{total_episodes})")
    
    env.close()
    eval_env.close()
