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
        
        # STL setup
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
            "models/IRB1100_xistera_right/urdf/IRB1100_xistera_right.urdf",
            [0, 0, 0.8],
            useFixedBase=1,
        )
        
        ee_min, ee_max = self._measure_workspace()
        # add a small margin
        margin = np.array([0.05, 0.05, 0.05])
        ee_low = ee_min - margin
        ee_high = ee_max + margin

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
            low=ee_low.astype(np.float32),
            high=ee_high.astype(np.float32),
            shape=(3,),
            dtype=np.float32
            ),
            'desired_goal': spaces.Box(
                low=ee_low.astype(np.float32),
                high=ee_high.astype(np.float32),
                shape=(3,),
                dtype=np.float32
            )
        })

        self.t = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.ball = None
        self.ball_caught = False
        
        # Goal-related variables
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

        # Initialize goal to ball's initial position
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
        self.desired_goal = np.array(ball_pos, dtype=np.float32)

        return self._get_observation(), {}

    def _spawn_new_ball(self):
        """Spawn a new ball with random velocity"""
        z_velocity = random.uniform(1, 2)
        y_velocity = random.uniform(-1, 1)
        x_velocity = random.uniform(-8, -4)
        self.ball = Ball((2, 0, 2), (x_velocity, y_velocity, z_velocity))
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

        # Get observation (this will update the goal to current ball position)
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
            "is_success": self.ball_caught  # Success = ball caught
        }
        
        return observation, reward, done, False, info

    def _get_observation(self):
        # Get joint angles
        joint_states = p.getJointStates(self.robotId, range(6))
        joint_angles = np.array([state[0] for state in joint_states], dtype=np.float32)
        
        # Get ball position
        if self.ball is not None and self.ball.id is not None:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball.id)
            ball_position = np.array(ball_pos, dtype=np.float32)
            
            # UPDATE: Set goal to current ball position (dynamic tracking)
            self.desired_goal = ball_position.copy()
        else:
            ball_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            # Keep previous goal if ball is gone
        
        # Get end-effector position (achieved goal)
        end_effector_state = p.getLinkState(self.robotId, 5)
        achieved_goal = np.array(end_effector_state[0], dtype=np.float32)
        
        # Combine joint angles and ball position for observation
        observation = np.concatenate([joint_angles, ball_position])
        
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': self.desired_goal.copy()
        }

    def _measure_workspace(self, n_samples=2000):
            """Randomly sample joint angles and record EE positions."""
            positions = []
            for _ in range(n_samples):
                # random joints in [-pi,pi]
                qs = np.random.uniform(-np.pi, np.pi, size=6)
                for j in range(6):
                    p.resetJointState(self.robotId, j, qs[j])
                ee_pos = p.getLinkState(self.robotId, 5)[0]
                positions.append(ee_pos)
            pos = np.array(positions)
            mins = pos.min(axis=0)
            maxs = pos.max(axis=0)
            return mins, maxs

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        SPARSE REWARD - Only success for ball catching:
        - 0.0 if ball is caught (contact with robot)
        - -1.0 otherwise (no reward for proximity)
        """
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        
        if achieved_goal.ndim == 1:
            # Single goal case
            
            # Check for ball catch - ONLY source of positive reward
            if self.ball is not None and self.ball.id is not None:
                contacts_ball_robot = p.getContactPoints(self.ball.id, self.robotId, -1, 5)  # 5 is the end effector link index
                if len(contacts_ball_robot) > 0:
                    self.ball_caught = True
                    return 0.0  # Success! Ball caught
            
            # No ball caught = failure
            return -1.0
        
        else:
            # Batch case (for HER experience replay)
            # For HER, we still need to check goal achievement for relabeled goals
            # But in practice, this will only be 0.0 when the relabeled goal 
            # corresponds to a position where the ball was actually caught
            distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            return np.where(distances <= self.goal_tolerance, 0.0, -1.0)

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

    def get_average_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
