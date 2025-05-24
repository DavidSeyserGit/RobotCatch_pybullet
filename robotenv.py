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
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[1, 0.75, 1],
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # Load plane and robot
        p.loadURDF("plane.urdf")
        
        # Example of loading an STL mesh with BOTH visual and collision shapes
        stl_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="models/station.STL",
            meshScale=[0.001, 0.001, 0.001],  # conversion from mm to m
        )
        stl_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="models/station.STL",
            meshScale=[0.001, 0.001, 0.001],
        )
        
        # Create a multi-body with BOTH visual and collision shapes
        self.stl_body_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseVisualShapeIndex=stl_visual_shape_id,
            baseCollisionShapeIndex=stl_collision_shape_id,  # Add collision shape
            basePosition=[0.75, 0.75, 0],  # Change position as needed
            baseOrientation=p.getQuaternionFromEuler(
                [np.pi / 2, 0, -np.pi / 2]
            ),  # 90 deg around X axis
        )

        self.robotId = p.loadURDF(
            "models/IRB1100_xistera/urdf/IRB1100_xistera.urdf",
            [0, 0, 0.8],
            useFixedBase=1,
        )

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        obs_low = np.concatenate([
            np.full(6, -np.pi),  # Joint angles
            np.array([-10.0, -10.0, -2.0])  # Ball position bounds (x, y, z)
        ])
        obs_high = np.concatenate([
            np.full(6, np.pi),   # Joint angles
            np.array([10.0, 10.0, 5.0])   # Ball position bounds (x, y, z)
        ])
        
        self.observation_space = spaces.Box(
            low=obs_low, 
            high=obs_high, 
            shape=(9,),  # 6 joint angles + 3 ball coordinates
            dtype=np.float32
        )

        self.t = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.previous_distance = None  # Initialize previous_distance
        self.ball = None  # Initialize ball object here
        self.ball_caught = False  # Track if ball was caught this episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.episode_reward = 0
        self.previous_distance = None  # Reset previous distance tracker
        self.ball_caught = False  # Reset ball caught flag
        
        # Reset the robot to its initial position
        for i in range(6):
            p.resetJointState(self.robotId, i, 0)

        # Remove any existing ball and spawn a new one
        if self.ball is not None:
            self.ball.remove()
        p.removeAllUserDebugItems()  # Clear debug items

        # Create a new ball with random velocity for each episode
        self._spawn_new_ball()

        # Return the initial observation
        return self._get_observation(), {}

    def _spawn_new_ball(self):
        """Spawn a new ball with random velocity"""
        z_velocity = random.uniform(1, 2)  # Random z velocity
        y_velocity = random.uniform(-0.5, 0.5)  # Random y velocity
        x_velocity = random.uniform(-8, -4)  # Random x velocity
        self.ball = Ball((3, 0, 1), (x_velocity, y_velocity, z_velocity))
        self.ball.spawn()
        self.ball.draw_velocity_vector()

    def step(self, action):
        # Scale actions from [-1, 1] to actual joint ranges
        scaled_action = np.interp(action, (-1, 1), (-np.pi, np.pi))

        # Apply actions to the robot joints
        for i in range(6):
            p.setJointMotorControl2(
                self.robotId, i, p.POSITION_CONTROL, targetPosition=scaled_action[i]
            )

        # Simulate the environment
        p.stepSimulation()

        # Get new observation (joint angles)
        observation = self._get_observation()

        # Calculate reward using the new reward function
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if the episode is done
        self.t += 1

        # Episode ends after 500 timesteps OR if ball is caught OR ball goes out of bounds
        done = (
            self.t >= 500 
            or self.ball_caught 
            or self._is_ball_out_of_bounds()
        )
        
        # Track rewards for the episode
        if done:
            self.episode_rewards.append(self.episode_reward)
            if self.ball is not None and self.ball.id is not None:
                self.ball.remove()
            p.removeAllUserDebugItems()

        info = {"ball_caught": self.ball_caught}
        # Return observation, reward, done flag, and info
        return observation, reward, done, False, info  # False represents 'not truncated'

    def _is_ball_out_of_bounds(self):
        """Check if ball has gone out of reasonable bounds"""
        if self.ball is None or self.ball.id is None:
            return True
        
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
        # Consider ball out of bounds if it's too far away or too low
        if (
            ball_pos[2] < -1  # Below ground significantly
            or abs(ball_pos[0]) > 10  # Too far in x direction
            or abs(ball_pos[1]) > 10  # Too far in y direction
        ):
            return True
        return False

    def close(self):
        p.disconnect()

    def _get_observation(self):
        # Get current joint angles
        joint_states = p.getJointStates(self.robotId, range(6))
        joint_angles = np.array([state[0] for state in joint_states], dtype=np.float32)
        
        # Get ball position
        if self.ball is not None and self.ball.id is not None:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
            ball_position = np.array(ball_pos, dtype=np.float32)
        else:
            # If no ball exists, use a default position (or zeros)
            ball_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Combine joint angles and ball position
        observation = np.concatenate([joint_angles, ball_position])
        return observation

    def _calculate_reward(self):
        """
        Calculates the reward based on collisions.
        - Positive reward for ball-robot collision.
        - Negative reward for robot-environment collision.
        """
        reward = 0

        # Check for collision between the ball and the robot
        contacts_ball_robot = p.getContactPoints(
            self.ball.id, self.robotId
        )  # ball id and robot id
        if len(contacts_ball_robot) > 0:
            reward += 100  # Substantial reward for collision
            self.ball_caught = True  # Mark ball as caught

        # Check for collision between the robot and the environment (excluding the ball)
        for body_id in range(
            p.getNumBodies()
        ):  # Iterate through all bodies in the environment
            if body_id != self.robotId and body_id != self.ball.id:
                contacts_robot_env = p.getContactPoints(
                    self.robotId, body_id
                )  # robot id and env id
                if len(contacts_robot_env) > 0:
                    reward -= 100  # Large penalty for collision with environment
                    break  # Only penalize once per step even if multiple collisions

        return reward

    def get_average_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
