#!/usr/bin/env python
"""
play_approx.py - Visualize trained Approximate Q-learning agent

Usage:
    python play_approx.py --layout classic
    python play_approx.py --layout classic --headless
    python play_approx.py --layout classic --scale 4 --speed 50
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import numpy as np

from pacman_env import PacmanEnv
from approx_qlearning import FeatureExtractor, ApproximateQLearningAgent

# Try to import cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - running in headless mode")


def is_display_available():
    """Check if display is available for OpenCV."""
    if not CV2_AVAILABLE:
        return False
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    try:
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False


def load_agent(layout: str, weight_path: Path | None = None) -> ApproximateQLearningAgent:
    """Load a trained agent."""
    # Create dummy env to get feature count
    env = PacmanEnv(layout)
    feature_extractor = FeatureExtractor(env)
    env.close()
    
    agent = ApproximateQLearningAgent(
        n_features=feature_extractor.n_features,
        n_actions=4,
        alpha=0.002,  # Match training params
        gamma=0.9,
    )
    
    weight_file = weight_path or Path(f"approx_q_weights_{layout}.csv")
    if weight_file.exists():
        agent.load(str(weight_file))
    else:
        print(f"Warning: No weights at {weight_file}, using untrained agent")
    
    return agent


def play_visual(layout: str, agent: ApproximateQLearningAgent, 
                episodes: int, delay_ms: int, scale: int):
    """Play with OpenCV visualization."""
    env = PacmanEnv(layout, reward_shaping=False)
    wins = 0
    
    cv2.namedWindow("Pac-Man (Approx Q)", cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)
    
    for ep in range(1, episodes + 1):
        env.reset()
        feature_extractor = FeatureExtractor(env)
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 3000:
            # Get action from agent
            action = agent.get_action(feature_extractor, training=False)
            
            # Show Q-values for debugging
            if step < 5 or step % 100 == 0:
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                q_vals = [agent.get_q_value(feature_extractor.extract(a)) for a in range(4)]
                print(f"  Step {step}: Action={actions[action]}, "
                      f"Q-values={[f'{q:.2f}' for q in q_vals]}, "
                      f"Pellets: {len(env.pellets)}")
            
            # Take action
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            
            if not done:
                feature_extractor = FeatureExtractor(env)
            
            # Render
            frame = cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR)
            if scale > 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
            
            # Add text overlay
            cv2.putText(frame, f"{layout} Ep {ep}/{episodes} Step {step} Pellets {len(env.pellets)}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Pac-Man (Approx Q)", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                print("Quit requested.")
                env.close()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord(' '):  # Pause on spacebar
                cv2.waitKey(0)
        
        win = done and not env.pellets
        wins += int(win)
        result = "WIN" if win else "LOSE"
        print(f"Episode {ep}: {result} in {step} steps, reward={total_reward:.1f}")
    
    print(f"\n{wins}/{episodes} wins ({wins/episodes*100:.1f}%)")
    env.close()
    cv2.destroyAllWindows()


def play_headless(layout: str, agent: ApproximateQLearningAgent, episodes: int):
    """Play without display."""
    env = PacmanEnv(layout, reward_shaping=False)
    wins = 0
    
    print(f"Running Approx Q-Learning agent on {layout} (headless)")
    print(f"Weights: {agent.weights}")
    print("-" * 60)
    
    for ep in range(1, episodes + 1):
        env.reset()
        feature_extractor = FeatureExtractor(env)
        
        done = False
        step = 0
        total_reward = 0
        initial_pellets = len(env.pellets)
        
        print(f"\nEpisode {ep}/{episodes}: {initial_pellets} pellets")
        
        while not done and step < 3000:
            action = agent.get_action(feature_extractor, training=False)
            
            if step < 3 or step % 200 == 0:
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                q_vals = [agent.get_q_value(feature_extractor.extract(a)) for a in range(4)]
                print(f"  Step {step}: {actions[action]}, Q={[f'{q:.1f}' for q in q_vals]}, "
                      f"Pos={env.pac_pos}, Pellets={len(env.pellets)}")
            
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            
            if not done:
                feature_extractor = FeatureExtractor(env)
        
        win = done and not env.pellets
        wins += int(win)
        collected = initial_pellets - len(env.pellets)
        result = "WIN" if win else "LOSE"
        print(f"Episode {ep}: {result} | Steps: {step} | "
              f"Collected: {collected}/{initial_pellets} | Reward: {total_reward:.1f}")
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {wins}/{episodes} wins ({wins/episodes*100:.1f}%)")
    print(f"{'='*60}")
    env.close()


def play(layout: str, agent: ApproximateQLearningAgent, episodes: int,
         delay_ms: int = 100, scale: int = 3, headless: bool = False):
    """Play with automatic display detection."""
    if headless or not is_display_available():
        if not headless:
            print("No display available - running headless")
        play_headless(layout, agent, episodes)
    else:
        play_visual(layout, agent, episodes, delay_ms, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Pac-Man with Approx Q-Learning")
    parser.add_argument("--layout", type=str, default="classic",
                        choices=["empty", "spiral", "spiral_harder", "classic"])
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to weights file (default: approx_q_weights_<layout>.csv)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--speed", type=int, default=100,
                        help="Delay between frames in ms")
    parser.add_argument("--scale", type=int, default=3,
                        help="Window scale factor")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    
    args = parser.parse_args()
    
    # Load agent
    agent = load_agent(args.layout, args.weights)
    
    # Play
    play(args.layout, agent, args.episodes, args.speed, args.scale, args.headless)
