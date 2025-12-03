#!/usr/bin/env python
"""
approx_qlearning.py - Feature-based Approximate Q-learning for Pac-Man

This approach learns in ~200-500 episodes instead of 30,000+ required for pixel-based DQN.
Based on the approach from https://github.com/tadowney/ai_pacman

Key insight: Instead of learning from pixels, we hand-craft meaningful features
that capture the essential state information.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple, List, Dict
import random
import csv
from pathlib import Path

from pacman_env import PacmanEnv, DIRS


class FeatureExtractor:
    """Extract meaningful features from the game state."""
    
    def __init__(self, env: PacmanEnv):
        self.env = env
        self.n_features = 8  # Number of features we extract
    
    def _bfs_distance(self, start: Tuple[int, int], targets: List[Tuple[int, int]], 
                      max_depth: int = 50) -> Tuple[int, Tuple[int, int] | None]:
        """
        BFS to find shortest path distance to nearest target.
        Returns (distance, target_position) or (max_depth, None) if not found.
        
        This is better than Manhattan distance because it respects walls!
        """
        if not targets:
            return max_depth, None
        
        if start in targets:
            return 0, start
            
        visited = {start}
        queue = deque([(start, 0)])
        
        while queue:
            pos, dist = queue.popleft()
            if dist >= max_depth:
                break
                
            for dx, dy in DIRS:
                nx, ny = pos[0] + dx, pos[1] + dy
                
                # Handle portal wraparound for classic layout
                if self.env.layout_name == "classic" and pos[0] == 7:  # Portal row
                    if ny < 0:
                        ny = self.env.w - 1
                    elif ny >= self.env.w:
                        ny = 0
                
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < self.env.h and 0 <= ny < self.env.w):
                    continue
                if self.env.floor[nx, ny] == 1:  # Wall
                    continue
                    
                if (nx, ny) in targets:
                    return dist + 1, (nx, ny)
                    
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        # Fallback to Manhattan distance if BFS fails
        if targets:
            min_dist = min(abs(start[0] - t[0]) + abs(start[1] - t[1]) for t in targets)
            return min(min_dist, max_depth), None
        return max_depth, None
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get legal neighboring positions."""
        neighbors = []
        for dx, dy in DIRS:
            nx, ny = pos[0] + dx, pos[1] + dy
            
            # Handle portal
            if self.env.layout_name == "classic" and pos[0] == 7:
                if ny < 0:
                    ny = self.env.w - 1
                elif ny >= self.env.w:
                    ny = 0
            
            if 0 <= nx < self.env.h and 0 <= ny < self.env.w:
                if self.env.floor[nx, ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors
    
    def _position_after_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get position after taking an action (accounting for walls)."""
        dx, dy = DIRS[action]
        nx, ny = pos[0] + dx, pos[1] + dy
        
        # Handle portal
        if self.env.layout_name == "classic" and pos[0] == 7:
            if ny < 0:
                ny = self.env.w - 1
            elif ny >= self.env.w:
                ny = 0
        
        # Clamp to bounds
        nx = max(0, min(nx, self.env.h - 1))
        ny = max(0, min(ny, self.env.w - 1))
        
        # Check wall
        if self.env.floor[nx, ny] == 1:
            return pos  # Can't move into wall
        return (nx, ny)
    
    def extract(self, action: int) -> np.ndarray:
        """
        Extract features for a (state, action) pair.
        
        Features:
        0. Bias term (always 1)
        1. Ghost 1 step away after action (binary) - DANGER!
        2. Ghost 2 steps away after action (binary) - WARNING
        3. Ghost 3+ steps away or no ghost (binary) - SAFE
        4. Eating food with this action (binary)
        5. 1 / (distance to closest food + 1) after action
        6. Moving toward food (binary)
        7. Number of legal moves from new position / 4 (avoid corners)
        """
        features = np.zeros(self.n_features)
        
        # Feature 0: Bias
        features[0] = 1.0
        
        # Get position after action
        current_pos = self.env.pac_pos
        new_pos = self._position_after_action(current_pos, action)
        
        # Features 1-3: Ghost proximity (using BFS for accuracy)
        ghost_dist, _ = self._bfs_distance(new_pos, self.env.ghost_pos, max_depth=10)
        
        features[1] = 1.0 if ghost_dist <= 1 else 0.0  # DANGER
        features[2] = 1.0 if ghost_dist == 2 else 0.0  # WARNING  
        features[3] = 1.0 if ghost_dist >= 3 else 0.0  # SAFE
        
        # Feature 4: Eating food
        features[4] = 1.0 if new_pos in self.env.pellets else 0.0
        
        # Feature 5: Inverse distance to closest food
        food_dist_after, _ = self._bfs_distance(new_pos, self.env.pellets, max_depth=30)
        features[5] = 1.0 / (food_dist_after + 1)
        
        # Feature 6: Moving toward food
        food_dist_before, _ = self._bfs_distance(current_pos, self.env.pellets, max_depth=30)
        features[6] = 1.0 if food_dist_after < food_dist_before else 0.0
        
        # Feature 7: Mobility (avoid getting cornered)
        neighbors = self._get_neighbors(new_pos)
        features[7] = len(neighbors) / 4.0
        
        return features


class ApproximateQLearningAgent:
    """
    Approximate Q-learning agent using linear function approximation.
    
    Q(s, a) = w · f(s, a)
    
    where w is a weight vector and f(s, a) are the features.
    """
    
    def __init__(self, n_features: int, n_actions: int = 4,
                 alpha: float = 0.002, gamma: float = 0.9,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 100):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate - MUCH SMALLER to prevent explosion
        self.gamma = gamma  # Discount factor - slightly lower for stability
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize weights to zeros (more stable than random)
        self.weights = np.zeros(n_features)
        
        # Weight clipping to prevent explosion
        self.max_weight = 100.0
        
        self.episode = 0
    
    @property
    def epsilon(self) -> float:
        """Current epsilon value with decay."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.episode / self.epsilon_decay)
    
    def get_q_value(self, features: np.ndarray) -> float:
        """Compute Q-value as dot product of weights and features."""
        return np.dot(self.weights, features)
    
    def get_action(self, feature_extractor: FeatureExtractor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        # Compute Q-values for all actions
        q_values = []
        for action in range(self.n_actions):
            features = feature_extractor.extract(action)
            q_values.append(self.get_q_value(features))
        
        # Return action with highest Q-value (break ties randomly)
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, features: np.ndarray, reward: float, 
               next_feature_extractor: FeatureExtractor | None, done: bool):
        """
        Update weights using TD learning with stability improvements.
        
        w  w + ± * (reward + ³ * max_a' Q(s', a') - Q(s, a)) * f(s, a)
        """
        current_q = self.get_q_value(features)
        
        if done or next_feature_extractor is None:
            target = reward
        else:
            # Compute max Q-value for next state
            next_q_values = []
            for action in range(self.n_actions):
                next_features = next_feature_extractor.extract(action)
                next_q_values.append(self.get_q_value(next_features))
            target = reward + self.gamma * max(next_q_values)
        
        # TD error with clipping to prevent explosion
        td_error = target - current_q
        td_error = np.clip(td_error, -10.0, 10.0)  # Clip TD error
        
        # Update weights
        self.weights += self.alpha * td_error * features
        
        # Clip weights to prevent explosion
        self.weights = np.clip(self.weights, -self.max_weight, self.max_weight)
    
    def save(self, path: str):
        """Save weights to CSV."""
        np.savetxt(path, self.weights, delimiter=',')
        print(f"Weights saved to {path}")
    
    def load(self, path: str):
        """Load weights from CSV."""
        self.weights = np.loadtxt(path, delimiter=',')
        print(f"Weights loaded from {path}")


def train(layout: str = "classic", episodes: int = 500, 
          verbose: bool = True) -> ApproximateQLearningAgent:
    """
    Train an approximate Q-learning agent.
    
    This should converge in ~200-500 episodes!
    """
    env = PacmanEnv(layout, reward_shaping=False)  # Use simple rewards
    
    # Create agent with stable hyperparameters
    feature_extractor = FeatureExtractor(env)
    agent = ApproximateQLearningAgent(
        n_features=feature_extractor.n_features,
        n_actions=4,
        alpha=0.002,  # Small learning rate for stability
        gamma=0.9,    # Lower discount for faster learning
        epsilon_decay=episodes // 3,  # Decay epsilon over first 33% of training
    )
    
    # Training stats
    rewards_history = []
    wins_history = []
    
    # Custom rewards - SCALED DOWN to prevent weight explosion
    # Keep rewards in a small range [-1, 1] approximately
    REWARD_FOOD = 1.0
    REWARD_WIN = 5.0
    REWARD_DEATH = -5.0
    REWARD_STEP = -0.01  # Tiny penalty to encourage efficiency
    
    for episode in range(1, episodes + 1):
        agent.episode = episode
        state_rgb, _ = env.reset()
        feature_extractor = FeatureExtractor(env)
        
        done = False
        total_reward = 0
        steps = 0
        max_steps = 3000  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Select action
            action = agent.get_action(feature_extractor, training=True)
            
            # Extract features BEFORE taking action
            features = feature_extractor.extract(action)
            
            # Store pre-action state
            had_pellet = env.pac_pos in env.pellets or \
                        feature_extractor._position_after_action(env.pac_pos, action) in env.pellets
            pellets_before = len(env.pellets)
            
            # Take action
            _, env_reward, done, _, _ = env.step(action)
            steps += 1
            
            # Compute our custom reward
            pellets_after = len(env.pellets)
            ate_food = pellets_after < pellets_before
            
            if done and not env.pellets:  # Win!
                reward = REWARD_WIN
            elif done:  # Death
                reward = REWARD_DEATH
            elif ate_food:
                reward = REWARD_FOOD
            else:
                reward = REWARD_STEP
            
            total_reward += reward
            
            # Update features for next state
            if not done:
                next_feature_extractor = FeatureExtractor(env)
            else:
                next_feature_extractor = None
            
            # Learn!
            agent.update(features, reward, next_feature_extractor, done)
            
            feature_extractor = next_feature_extractor
        
        # Track stats
        win = 1 if (done and not env.pellets) else 0
        rewards_history.append(total_reward)
        wins_history.append(win)
        
        # Print progress
        if verbose and (episode % 50 == 0 or episode <= 10):
            window = min(50, len(rewards_history))
            avg_reward = np.mean(rewards_history[-window:])
            avg_win = np.mean(wins_history[-window:]) * 100
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Steps: {steps:4d} | "
                  f"Win: {win} | "
                  f"Avg Reward (last {window}): {avg_reward:7.2f} | "
                  f"Win Rate: {avg_win:5.1f}% | "
                  f"µ: {agent.epsilon:.3f}")
            # Format weights nicely
            w_str = ", ".join([f"{w:6.2f}" for w in agent.weights])
            print(f"  Weights: [{w_str}]")
    
    env.close()
    
    # Final stats
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final weights: {agent.weights}")
    print(f"Final win rate (last 50): {np.mean(wins_history[-50:])*100:.1f}%")
    print(f"Final avg reward (last 50): {np.mean(rewards_history[-50:]):.1f}")
    
    # Save weights
    weight_path = f"approx_q_weights_{layout}.csv"
    agent.save(weight_path)
    
    # Save training log
    log_path = f"approx_q_log_{layout}.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'win'])
        for i, (r, w) in enumerate(zip(rewards_history, wins_history)):
            writer.writerow([i+1, r, w])
    print(f"Training log saved to {log_path}")
    
    return agent


def test(layout: str = "classic", episodes: int = 50, 
         weight_path: str | None = None, verbose: bool = True) -> float:
    """Test a trained agent."""
    env = PacmanEnv(layout, reward_shaping=False)
    
    # Create agent and load weights
    feature_extractor = FeatureExtractor(env)
    agent = ApproximateQLearningAgent(
        n_features=feature_extractor.n_features,
        n_actions=4,
        alpha=0.002,  # Match training params
        gamma=0.9,
    )
    
    weight_path = weight_path or f"approx_q_weights_{layout}.csv"
    if Path(weight_path).exists():
        agent.load(weight_path)
    else:
        print(f"Warning: No weights found at {weight_path}, using random weights")
    
    wins = 0
    total_rewards = []
    
    for episode in range(1, episodes + 1):
        state_rgb, _ = env.reset()
        feature_extractor = FeatureExtractor(env)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 3000:
            action = agent.get_action(feature_extractor, training=False)
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if not done:
                feature_extractor = FeatureExtractor(env)
        
        win = 1 if (done and not env.pellets) else 0
        wins += win
        total_rewards.append(total_reward)
        
        if verbose:
            result = "WIN" if win else "LOSE"
            print(f"Episode {episode}: {result} | Steps: {steps} | Reward: {total_reward:.1f}")
    
    win_rate = wins / episodes * 100
    print(f"\nTest Results: {wins}/{episodes} wins ({win_rate:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    
    env.close()
    return win_rate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Approximate Q-learning for Pac-Man")
    parser.add_argument("--layout", type=str, default="classic",
                        choices=["empty", "spiral", "spiral_harder", "classic"])
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "both"])
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--test-episodes", type=int, default=50,
                        help="Number of test episodes")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        print(f"Training on {args.layout} for {args.episodes} episodes...")
        agent = train(args.layout, args.episodes)
    
    if args.mode in ["test", "both"]:
        print(f"\nTesting on {args.layout} for {args.test_episodes} episodes...")
        test(args.layout, args.test_episodes)