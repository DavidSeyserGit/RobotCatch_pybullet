import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from ball import Ball, Simulation
import random
import os


class HERRobotEnv(gym.Env):
    def __init__(self):
        super(HERRobotEnv, self).__init__()
        # Curriculum learning parameters
        self.curriculum_phase = 0  # Starts at easiest phase
        self.success_window = []  # Track recent successes
        self.window_size = 100  # Number of episodes to consider
        self.promotion_threshold = 0.2  # 20% success rate to increase difficulty
        
        # Phase descriptions for logging
        self.phase_descriptions = [
            "Phase 0: Close range (1.0-1.5m), Slow balls (-3 to -2 m/s)",
            "Phase 1: Medium range (1.3-1.8m), Medium speed (-4 to -3 m/s)",
            "Phase 2: Long range (1.5-2.0m), Fast balls (-6 to -4 m/s)",
            "Phase 3: Full range (1.5-2.5m), Full speed (-8 to -4 m/s)"
        ]
        print(f"\nStarting curriculum learning at {self.phase_descriptions[0]}")
        
        # PyBullet setup
        self.physicsClient = p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[1, 0.75, 1],
        )
        # Add both pybullet_data and our models directory to the search path
        pybullet_data_path = pybullet_data.getDataPath()
        p.setAdditionalSearchPath(pybullet_data_path)
        models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        p.setAdditionalSearchPath(models_path)
        p.setGravity(0, 0, -9.81)
        
        # Load plane and robot with absolute paths
        plane_urdf = os.path.join(pybullet_data_path, "plane.urdf")
        p.loadURDF(plane_urdf)
        
        # STL setup - now using just the filename since we added the models path
        try:
            print(f"Attempting to load station STL from search paths...")
            print(f"Current search paths:")
            for path in [pybullet_data_path, models_path]:
                print(f"- {path}")
            
            # First try with absolute path
            stl_path = os.path.join(models_path, "station.STL")
            print(f"Trying absolute path: {stl_path}")
            
            stl_visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=stl_path,
                meshScale=[0.001, 0.001, 0.001],
                rgbaColor=[0.7, 0.7, 0.7, 1],  # Light gray color
            )
            if stl_visual_shape_id < 0:
                raise ValueError("Failed to create visual shape")
                
            stl_collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=stl_path,
                meshScale=[0.001, 0.001, 0.001],
            )
            if stl_collision_shape_id < 0:
                raise ValueError("Failed to create collision shape")
            
            print("Successfully created visual and collision shapes")
            
            self.stl_body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=stl_visual_shape_id,
                baseCollisionShapeIndex=stl_collision_shape_id,
                basePosition=[0.75, 0.75, 0],
                baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, -np.pi / 2]),
            )
            if self.stl_body_id < 0:
                raise ValueError("Failed to create multibody")
                
            print(f"Successfully created station with body ID: {self.stl_body_id}")
            
        except Exception as e:
            print(f"Error loading station STL: {str(e)}")
            print("Continuing without station...")
            self.stl_body_id = -1

        # Load robot with absolute path
        robot_urdf = os.path.join(models_path, "IRB1100_xistera_right/urdf/IRB1100_xistera_right.urdf")
        self.robotId = p.loadURDF(
            robot_urdf,
            [0, 0, 0.8],
            useFixedBase=1,
        )

        # Define static workspace bounds (in meters)
        ee_low = np.array([-0.8, -0.8, 0.0])  # Minimum reach
        ee_high = np.array([0.8, 0.8, 1.6])   # Maximum reach

        # Define action space (6 joint angles)
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
        """Spawn a new ball with random position and velocity based on curriculum phase"""
        # Curriculum-based parameters for distance from robot
        distance_ranges = [
            (2.0, 2.5),   # Phase 0: Close range
            (2.3, 2.8),   # Phase 1: Medium range
            (2.5, 3.0),   # Phase 2: Longer range
            (2.5, 3.5),   # Phase 3: Full range
        ]
        velocity_ranges = [
            (-5, -4),     # Phase 0: Slow
            (-7, -5),     # Phase 1: Medium
            (-8, -6),     # Phase 2: Fast
            (-10, -6),     # Phase 3: Full speed
        ]
        # Get current ranges
        phase = min(self.curriculum_phase, len(distance_ranges) - 1)
        dist_range = distance_ranges[phase]
        vel_range = velocity_ranges[phase]
        
        # Robot's position (center point for aiming)
        robot_pos = np.array([0.0, 0.0, 0.8])  # Robot base position
        target_pos = np.array([0.0, 0.0, 2])  # Aim slightly above robot base
        
        # Random angle from which to throw the ball (restricted to face robot)
        angle = random.uniform(-np.pi/7, np.pi/7)  # ±30 degrees
        
        # Calculate initial position based on angle and distance
        distance = random.uniform(*dist_range)
        x_pos = distance * np.cos(angle)
        y_pos = distance * np.sin(angle)
        z_pos = random.uniform(1.6, 2.0)     # Height range
        
        # Calculate velocity vector that aims towards the target position
        start_pos = np.array([x_pos, y_pos, z_pos])
        direction = target_pos - start_pos
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Base velocity magnitude
        velocity_magnitude = abs(random.uniform(*vel_range))
        
        # Add some randomness to velocity while maintaining general direction
        velocity_variation = 0.2  # 20% variation
        x_velocity = velocity_magnitude * direction[0] * (1 + random.uniform(-velocity_variation, velocity_variation))
        y_velocity = velocity_magnitude * direction[1] * (1 + random.uniform(-velocity_variation, velocity_variation))
        z_velocity = random.uniform(0.5, 1.5)  # Upward velocity component
        
        self.ball = Ball((x_pos, y_pos, z_pos), (x_velocity, y_velocity, z_velocity))
        self.ball.spawn()
        self.ball.draw_velocity_vector()

    def _update_curriculum(self, success):
        """Update curriculum phase based on recent performance"""
        self.success_window.append(float(success))
        if len(self.success_window) > self.window_size:
            self.success_window.pop(0)
            
        # Calculate success rate over window
        if len(self.success_window) == self.window_size:
            success_rate = np.mean(self.success_window)
            
            # Promote to next phase if doing well
            if success_rate >= self.promotion_threshold and self.curriculum_phase < 3:
                self.curriculum_phase += 1
                print(f"\nCurriculum advanced! Success rate: {success_rate*100:.1f}%")
                print(f"Now training at {self.phase_descriptions[self.curriculum_phase]}")
                self.success_window = []  # Reset window after promotion
            else:
                # Periodically report progress in current phase
                print(f"\nCurrent phase {self.curriculum_phase} progress:")
                print(f"Success rate: {success_rate*100:.1f}% (need {self.promotion_threshold*100:.1f}% to advance)")
                print(f"Remaining in {self.phase_descriptions[self.curriculum_phase]}")

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
            # Update curriculum based on episode success
            self._update_curriculum(self.ball_caught)
            
            self.episode_rewards.append(self.episode_reward)
            if self.ball is not None and self.ball.id is not None:
                self.ball.remove()
            p.removeAllUserDebugItems()

        # Add is_success to info for HER
        info = {
            "ball_caught": self.ball_caught,
            "is_success": self.ball_caught,  # Success = ball caught
            "curriculum_phase": self.curriculum_phase  # Add phase to info
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

    def _check_collisions(self):
        """Check for robot self-collisions and station collisions"""
        # Check self collisions between robot links
        for link1 in range(p.getNumJoints(self.robotId)):
            for link2 in range(link1 + 2, p.getNumJoints(self.robotId)):  # Skip adjacent links
                if p.getContactPoints(self.robotId, self.robotId, link1, link2):
                    return True, "self"
                    
        # Check station collisions if station exists
        if self.stl_body_id >= 0:
            for link_id in range(p.getNumJoints(self.robotId)):
                if p.getContactPoints(self.robotId, self.stl_body_id, link_id):
                    return True, "station"
        
        return False, None

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        SPARSE REWARD:
        - 0.0 if ball is caught (contact with robot)
        - -1.0 if collision with station or self
        - -1.0 otherwise (no reward for proximity)
        """
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        
        if achieved_goal.ndim == 1:
            # Single goal case
            
            # First check for collisions
            has_collision, _ = self._check_collisions()
            if has_collision:
                return -1.0  # Collision penalty
            
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
            or abs(ball_pos[0]) > 3
            or abs(ball_pos[1]) > 3
        ):
            return True
        return False

    def close(self):
        try:
            if p.isConnected(physicsClientId=self.physicsClient):
                p.disconnect(physicsClientId=self.physicsClient)
        except Exception as e:
            print(f"Warning during environment cleanup: {e}")
            # Continue gracefully even if there's an error

    def get_average_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
