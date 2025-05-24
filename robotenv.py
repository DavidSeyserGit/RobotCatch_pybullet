import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from ball import Ball, Simulation
import random


class HERRobotEnv(gym.Env):
    def __init__(self):
        super(HERRobotEnv, self).__init__()
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
        
        # STL setup (same as before)
        stl_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="models/station.STL",
            meshScale=[0.001, 0.001, 0.001],
        )
        stl_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="models/station.STL",
            meshScale=[0.001, 0.001, 0.001],
        )
        
        self.stl_body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=stl_visual_shape_id,
            baseCollisionShapeIndex=stl_collision_shape_id,
            basePosition=[0.75, 0.75, 0],
            baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, -np.pi / 2]),
        )

        self.robotId = p.loadURDF(
            "models/IRB1100_xistera/urdf/IRB1100_xistera.urdf",
            [0, 0, 0.8],
            useFixedBase=1,
        )

        # Define action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # Define HER-compatible observation space (Dict format)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=np.concatenate([
                    np.full(6, -np.pi),  # Joint angles
                    np.array([-10.0, -10.0, -2.0])  # Ball position
                ]),
                high=np.concatenate([
                    np.full(6, np.pi),
                    np.array([10.0, 10.0, 5.0])
                ]),
                shape=(9,),
                dtype=np.float32
            ),
            'achieved_goal': spaces.Box(
                low=np.array([-3.0, -3.0, 0.0]),  # End-effector position bounds
                high=np.array([3.0, 3.0, 3.0]),
                shape=(3,),
                dtype=np.float32
            ),
            'desired_goal': spaces.Box(
                low=np.array([-3.0, -3.0, 0.0]),  # Goal position bounds
                high=np.array([3.0, 3.0, 3.0]),
                shape=(3,),
                dtype=np.float32
            )
        })

        self.t = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.ball = None
        self.ball_caught = False
        
        # Add goal-related variables
        self.desired_goal = None
        self.goal_tolerance = 0.15  # Distance threshold for success (15cm)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.episode_reward = 0
        self.ball_caught = False
        
        # Reset robot
        for i in range(6):
            p.resetJointState(self.robotId, i, 0)

        # Remove existing ball and spawn new one
        if self.ball is not None:
            self.ball.remove()
        p.removeAllUserDebugItems()
        
        self._spawn_new_ball()

        # Set goal as ball's initial position (where we want to intercept)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
        self.desired_goal = np.array(ball_pos, dtype=np.float32)

        # Return HER-compatible observation
        return self._get_observation(), {}

    def _spawn_new_ball(self):
        """Spawn a new ball with random velocity"""
        z_velocity = random.uniform(1, 2)
        y_velocity = random.uniform(-0.5, 0.5)
        x_velocity = random.uniform(-8, -4)
        self.ball = Ball((3, 0, 1), (x_velocity, y_velocity, z_velocity))
        self.ball.spawn()
        self.ball.draw_velocity_vector()

    def step(self, action):
        # Scale actions
        scaled_action = np.interp(action, (-1, 1), (-np.pi, np.pi))

        # Apply actions
        for i in range(6):
            p.setJointMotorControl2(
                self.robotId, i, p.POSITION_CONTROL, targetPosition=scaled_action[i]
            )

        p.stepSimulation()

        # Get observation
        observation = self._get_observation()

        # Use goal-conditioned reward
        reward = self.compute_reward(
            observation['achieved_goal'], 
            observation['desired_goal'], 
            {}
        )
        
        self.episode_reward += reward
        self.t += 1

        # Check if done
        done = (
            self.t >= 500 
            or self.ball_caught 
            or self._is_ball_out_of_bounds()
        )
        
        if done:
            self.episode_rewards.append(self.episode_reward)
            if self.ball is not None and self.ball.id is not None:
                self.ball.remove()
            p.removeAllUserDebugItems()

        # Add is_success to info for HER
        info = {
            "ball_caught": self.ball_caught,
            "is_success": self._is_success(
                observation['achieved_goal'], 
                observation['desired_goal']
            )
        }
        
        return observation, reward, done, False, info

    def _get_observation(self):
        # Return Dict observation for HER
        # Get joint angles
        joint_states = p.getJointStates(self.robotId, range(6))
        joint_angles = np.array([state[0] for state in joint_states], dtype=np.float32)
        
        # Get ball position
        if self.ball is not None and self.ball.id is not None:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
            ball_position = np.array(ball_pos, dtype=np.float32)
        else:
            ball_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Get end-effector position (achieved goal)
        end_effector_state = p.getLinkState(self.robotId, 5)  # Assuming link 5 is end-effector
        achieved_goal = np.array(end_effector_state[0], dtype=np.float32)
        
        # Combine joint angles and ball position for observation
        observation = np.concatenate([joint_angles, ball_position])
        
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': self.desired_goal.copy()
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        FIXED: Vectorized goal-conditioned reward function for HER.
        Handles both single goals and batches of goals.
        """
        # Convert to numpy arrays
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        
        # Handle both single goals and batches
        if achieved_goal.ndim == 1:
            # Single goal case
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -1.0 if distance > self.goal_tolerance else 0.0
        else:
            # Batch case (multiple goals)
            distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            rewards = np.where(distances > self.goal_tolerance, -1.0, 0.0)
            return rewards

    def _is_success(self, achieved_goal, desired_goal):
        """Check if the goal has been achieved"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self.goal_tolerance

    def _is_ball_out_of_bounds(self):
        """Check if ball has gone out of reasonable bounds"""
        if self.ball is None or self.ball.id is None:
            return True
        
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
        if (
            ball_pos[2] < -1
            or abs(ball_pos[0]) > 10
            or abs(ball_pos[1]) > 10
        ):
            return True
        return False

    def close(self):
        p.disconnect()

    # Keep your old reward function for reference, but it's not used in HER
    def _calculate_reward(self):
        """Original reward function (not used in HER)"""
        reward = 0
        contacts_ball_robot = p.getContactPoints(self.ball.id, self.robotId)
        if len(contacts_ball_robot) > 0:
            reward += 100
            self.ball_caught = True

        for body_id in range(p.getNumBodies()):
            if body_id != self.robotId and body_id != self.ball.id:
                contacts_robot_env = p.getContactPoints(self.robotId, body_id)
                if len(contacts_robot_env) > 0:
                    reward -= 100
                    break
        return reward

    def get_average_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
