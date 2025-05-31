# ppo_finetune_simple.py

import numpy as np
import gymnasium as gym
import argparse
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robotenv import HERRobotEnv

def make_env():
    """Constructs one Monitor-wrapped RobotEnv."""
    def _init():
        env = HERRobotEnv()
        return Monitor(env)
    return _init

def evaluate_model(model, env, n_episodes=3):
    """Evaluate a model for n episodes and return average reward."""
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()  # VecEnv reset() returns just the obs
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  # VecEnv step() returns (obs, reward, done, info)
            total_reward += reward[0]  # Extract scalar reward from array
            done = done[0]  # Extract scalar done from array
            truncated = info[0].get('TimeLimit.truncated', False)  # Check for truncation in info
        rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    print(f"Average reward: {avg_reward:.2f}")
    return avg_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="reinforcement_learning/her_sac_robot.zip",
                      help="Path to SAC model to load")
    parser.add_argument("--eval-only", action="store_true",
                      help="Only evaluate the model, don't train")
    parser.add_argument("--episodes", type=int, default=3,
                      help="Number of episodes for evaluation")
    args = parser.parse_args()

    # 1) build a single-env vector wrapper with normalization
    env = DummyVecEnv([make_env()])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,  # Keep sparse reward untouched
        clip_obs=10.0,
        training=False  # Don't update stats during evaluation
    )

    # 2) load your SAC+HER model
    print(f"Loading SAC model from {args.model}")
    sac = SAC.load(args.model, env=env)

    if args.eval_only:
        evaluate_model(sac, env, args.episodes)
        env.close()
        return

    # 3) extract its policy architecture and actor weights
    # Only keep the network architecture from SAC's policy_kwargs
    policy_kwargs = {
        'net_arch': sac.policy_kwargs.get('net_arch', [128, 128]),
        'activation_fn': sac.policy_kwargs.get('activation_fn', None)
    }
    actor_dict = sac.policy.state_dict()

    # 4) build a new PPO with MultiInputPolicy for dict observations
    ppo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./ppo_simple_tb/",
        verbose=1,
    )

    # 5) copy over the SAC actor weights into PPO's policy network
    # Only copy the actor part, not the critic
    for name, param in actor_dict.items():
        if 'actor' in name:  # Only copy actor parameters
            try:
                ppo.policy.state_dict()[name].copy_(param)
                print(f"Copied parameter: {name}")
            except KeyError:
                print(f"Skipping parameter: {name} (not found in PPO policy)")

    # 6) train for ~100 episodes = 100 * 500 steps ≈ 50 000 timesteps
    total_timesteps = 100 * 500
    ppo.learn(total_timesteps=total_timesteps)

    # 7) save your fine-tuned PPO
    ppo.save("ppo_robot_finetuned_simple.zip")

    # 8) Final evaluation
    print("\nFinal PPO evaluation:")
    evaluate_model(ppo, env, args.episodes)

    env.close()

if __name__ == "__main__":
    main()
