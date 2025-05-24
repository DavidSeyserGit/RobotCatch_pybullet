import numpy as np
from collections import deque
import random


class HERReplayBuffer:
    def __init__(self, capacity, k=4, strategy='future'):
        """
        HER Replay Buffer
        
        Args:
            capacity: Maximum number of transitions to store
            k: Number of additional goals to sample per transition
            strategy: 'future', 'episode', or 'random'
        """
        self.capacity = capacity
        self.k = k
        self.strategy = strategy
        self.buffer = deque(maxlen=capacity)
        self.episodes = []  # Store complete episodes for HER
        
    def add_episode(self, episode):
        """
        Add a complete episode to the buffer
        
        Args:
            episode: List of (obs, action, reward, next_obs, done, info) tuples
        """
        self.episodes.append(episode)
        
        # Add original transitions
        for transition in episode:
            self.buffer.append(transition)
        
        # Generate HER transitions
        her_transitions = self._generate_her_transitions(episode)
        for transition in her_transitions:
            self.buffer.append(transition)
    
    def _generate_her_transitions(self, episode):
        """Generate HER transitions by relabeling goals"""
        her_transitions = []
        
        for i, (obs, action, reward, next_obs, done, info) in enumerate(episode):
            # Sample k additional goals
            for _ in range(self.k):
                if self.strategy == 'future':
                    # Sample goal from future states in this episode
                    if i < len(episode) - 1:
                        future_idx = random.randint(i + 1, len(episode) - 1)
                        new_goal = episode[future_idx][3]['achieved_goal']  # next_obs achieved_goal
                    else:
                        continue
                elif self.strategy == 'episode':
                    # Sample goal from any state in this episode
                    goal_idx = random.randint(0, len(episode) - 1)
                    new_goal = episode[goal_idx][3]['achieved_goal']
                elif self.strategy == 'random':
                    # Sample random goal (you'd need to implement this)
                    new_goal = self._sample_random_goal()
                else:
                    continue
                
                # Create new transition with relabeled goal
                new_obs = obs.copy()
                new_obs['desired_goal'] = new_goal
                
                new_next_obs = next_obs.copy()
                new_next_obs['desired_goal'] = new_goal
                
                # Compute new reward
                new_reward = self._compute_reward(
                    new_next_obs['achieved_goal'], 
                    new_goal
                )
                
                # Check if this transition is successful with new goal
                new_info = info.copy()
                new_info['is_success'] = self._is_success(
                    new_next_obs['achieved_goal'], 
                    new_goal
                )
                
                her_transitions.append((
                    new_obs, action, new_reward, new_next_obs, done, new_info
                ))
        
        return her_transitions
    
    def _compute_reward(self, achieved_goal, desired_goal):
        """Compute reward for HER transitions"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -(distance > 0.1).astype(np.float32)
    
    def _is_success(self, achieved_goal, desired_goal):
        """Check success for HER transitions"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < 0.1
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for obs, action, reward, next_obs, done, info in batch:
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)
