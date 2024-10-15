import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from ball import Ball, Simulation
import random

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        # PyBullet setup
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # Load plane and robot
        p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("models/abb_irb120_support/urdf/abbIrb120.urdf", [0, 0, 0], useFixedBase=1)
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        self.target_pos = [0.5, 0.5, 0.5]
        self.t = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.previous_distance = None  # Initialize previous_distance
        self.ball = None  # Initialize ball object here
        self.episode_rewards = []  # Store rewards for the current episode
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.episode_reward = 0
        self.previous_distance = None  # Reset previous distance tracker
        # Reset the robot to its initial position
        for i in range(6):
            p.resetJointState(self.robotId, i, 0)
        
        # Remove any existing ball and spawn a new one
        if self.ball is not None:
            self.ball.remove()
        
        # Create a new ball with random velocity for each episode
        z_velocity = random.uniform(1, 2)  # Random z velocity
        y_velocity = random.uniform(-0.5, 0.5)  # Random y velocity
        x_velocity = random.uniform(-8, -4)  # Random x velocity
        self.ball = Ball((3, 0, 1),(x_velocity, y_velocity, z_velocity))
        self.ball.spawn()
        self.ball.draw_velocity_vector()
        
        # Return the initial observation
        return self._get_observation(), {}

    def step(self, action):
        # Scale actions from [-1, 1] to actual joint ranges
        scaled_action = np.interp(action, (-1, 1), (-np.pi, np.pi))

        # Apply actions to the robot joints
        for i in range(6):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=scaled_action[i])
        
        # Simulate the environment
        p.stepSimulation()
        
        # Get new observation (joint angles)
        observation = self._get_observation()
        
        # Calculate reward using the new reward function
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if the episode is done
        self.t += 1
        
        done = self.t >= 500  # Episode ends after 100 timesteps
        # Track rewards for the episode
        if done:
            self.episode_rewards.append(self.episode_reward)
            if self.ball is not None and self.ball.id is not None:
                self.ball.remove() 
                p.removeAllUserDebugItems()
        
        info = {}
        # Return observation, reward, done flag, and info
        return observation, reward, done, False, info  # False represents 'not truncated'

    def render(self):
        # PyBullet already renders if we use GUI
        pass

    def close(self):
        p.disconnect()

    def _get_observation(self):
        # Get current joint angles
        joint_states = p.getJointStates(self.robotId, range(6))
        return np.array([state[0] for state in joint_states], dtype=np.float32)

    def _calculate_reward(self):
        # Get end-effector position and calculate distance to the target
        end_effector_pos = np.array(p.getLinkState(self.robotId, 5)[0])
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_pos))
        
        # Base reward: negative distance to the target
        reward = -distance  # Scaling factor to moderate the impact of distance

        # Reward for getting closer to the target compared to the previous step
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - distance
            if distance_improvement > 0:
                reward += distance_improvement * 0.5  # Reward for getting closer
            else:
                reward -= 0.1  # Small penalty for getting further away
        
        # Set the previous distance for the next step
        self.previous_distance = distance

        # Bonus for getting very close to the target
        if distance < 0.05:
            reward += 20  # Increase the bonus for reaching the goal
        
        # Penalty for excessive movement (based on joint velocities)
        joint_velocities = np.array([state[1] for state in p.getJointStates(self.robotId, range(6))])
        movement_penalty = 0.001 * np.sum(np.abs(joint_velocities))  # Small penalty for joint movement
        reward -= movement_penalty

        return reward
    
    def get_average_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
