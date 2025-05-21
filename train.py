import gymnasium as gym
import numpy as np
import torch
# Assuming your RobotEnv class is defined in a file named robotenv.py
# And RobotEnv.__init__ now contains p.connect(p.DIRECT)
from robotenv import RobotEnv
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.env_checker import check_env
# Import SubprocVecEnv for training and DummyVecEnv for evaluation/visualization
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import pybullet as p # Import pybullet for manual GUI connection

class RewardThresholdCallback(BaseCallback):
    """
    Callback to stop training once the average reward over a check_freq
    reaches a specified threshold.
    """
    def __init__(self, reward_threshold, check_freq=1000, verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.total_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals['rewards']) > 0:
            self.total_rewards.append(np.mean(self.locals['rewards']))

        if self.n_calls % self.check_freq == 0:
            if len(self.total_rewards) >= self.check_freq:
                 average_reward = np.mean(self.total_rewards[-min(self.check_freq, len(self.total_rewards)):])
                 if self.verbose > 0:
                     print(f"Step: {self.num_timesteps}, Average reward over last {self.check_freq} steps: {average_reward}")

                 if average_reward >= self.reward_threshold:
                     return False
            elif self.verbose > 0 and len(self.total_rewards) > 0:
                 print(f"Step: {self.num_timesteps}, Accumulated average reward: {np.mean(self.total_rewards)}")

        return True

# make_env function remains simple, creating RobotEnv directly
# Assuming RobotEnv.__init__ now has p.connect(p.DIRECT)
def make_env():
    """
    Helper function to create a wrapped environment.
    RobotEnv handles its own PyBullet connection internally (assumed p.DIRECT).
    """
    def _init():
        # Create an instance of your RobotEnv
        env = RobotEnv()
        # Wrap the environment with Monitor to record rewards and episode lengths
        return Monitor(env)
    return _init

if __name__ == '__main__':
    # Number of parallel environments for training
    num_train_envs = 7 # You can change this for parallel training

    # Create the training environment using SubprocVecEnv for parallelism.
    # These environments will connect in p.DIRECT mode because RobotEnv.__init__() does.
    train_env = SubprocVecEnv([make_env() for _ in range(num_train_envs)])

    # Create a separate environment for evaluation.
    # We use DummyVecEnv here. This environment will also connect in p.DIRECT mode.
    eval_env = DummyVecEnv([make_env()])

    # Optional: Create a single environment for visualization with a GUI
    # This environment is NOT used for training or evaluation by Stable-Baselines3
    # and is only for manual stepping/rendering when needed.
    try:
        # Attempt to connect with GUI. This might fail if a GUI is already open.
        visual_client = p.connect(p.GUI)
        if visual_client >= 0:
            print("GUI connected for visualization.")
            # Create a separate environment instance using this client
            # You might need to modify your RobotEnv to accept a pre-connected client
            # if you want to use this approach directly.
            # For simplicity now, let's just note that you can use this client
            # to load models and step the simulation manually for visualization.
            # A more robust way would be to have a separate visualization function
            # that uses this client.
            # visual_env = RobotEnv(connection_mode=p.GUI) # This would work if RobotEnv accepted connection_mode
            pass # Placeholder if you don't modify RobotEnv for pre-connected client

        else:
            print("Could not connect to GUI. Visualization may not be available.")
            visual_client = None # Ensure visual_client is None if connection fails

    except Exception as e:
        print(f"Error connecting to GUI for visualization: {e}")
        visual_client = None


    # Setup EvalCallback to evaluate the model periodically on the eval_env (in DIRECT mode)
    # Rendering during EvalCallback will likely NOT show a GUI unless you
    # explicitly enable rendering in your RobotEnv's step/render methods AND
    # the PyBullet client supports it in DIRECT mode (which it usually doesn't for GUI).
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False) # Set render=False here


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_robot_tensorboard/", device=device)

    reward_threshold = 100
    reward_callback = RewardThresholdCallback(reward_threshold=reward_threshold, check_freq=1000)

    print("Starting training...")
    model.learn(total_timesteps=100000, callback=[eval_callback, reward_callback])

    model.save("ppo_robot_final")
    print("Training finished. Model saved as ppo_robot_final.zip")

    # *** How to Visualize ***
    # You have a few options for visualization now:

    # Option 1: Manually step the visualization client (most control)
    if visual_client is not None:
        print("Manually testing trained model with GUI...")
        # You would need to manually create a RobotEnv instance using visual_client
        # or load your robot/environment directly using the visual_client API.
        # This requires more manual PyBullet code here or modifying RobotEnv.
        # Example (requires RobotEnv to accept client or replicate its setup):
        # visual_robot_id = p.loadURDF("models/IRB1100_xistera/urdf/IRB1100_xistera.urdf", [0, 0, 0], useFixedBase=1, physicsClient=visual_client)
        # ... reset joint states, create ball, etc. using visual_client and visual_robot_id ...
        # for _ in range(1000):
        #     obs_manual = get_observation_manual(visual_client, visual_robot_id) # You need a function for this
        #     action_manual, _ = model.predict(obs_manual, deterministic=True)
        #     set_joint_commands_manual(visual_client, visual_robot_id, action_manual) # You need a function for this
        #     p.stepSimulation(physicsClient=visual_client)
        #     time.sleep(1./240.) # Optional: slow down for viewing

        # A simpler way if you just want to see the final behavior:
        # You could load the trained model in a separate script that
        # initializes ONE environment with p.GUI and runs the inference loop.
        pass # Placeholder


    # Option 2: Rely on the EvalCallback's render (if you managed to make it work with DIRECT or in a separate GUI process)
    # This is less reliable with the current PyBullet/macOS GUI interaction in multiprocessing.

    # Closing the DIRECT environments
    train_env.close()
    eval_env.close()

    # Manually disconnect the GUI client if it was connected
    if visual_client is not None and p.isConnected(visual_client):
        p.disconnect(visual_client)

    print("Environments closed.")

