#!/usr/bin/env python
"""
play_cv.py - Visualize and evaluate trained Pac-Man DQN agent

Usage examples:
    python play_cv.py --layout classic --episodes 20
    python play_cv.py --layout classic --headless
    python play_cv.py --layout classic --scale 4 --speed 50
    python play_cv.py --eval-all --model pacman_dqn_multitask.pt
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

from pacman_env import PacmanEnv
from dqn_agent import DQN, FrameStack, DEVICE

# Try to import cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - running in headless mode")

# Configuration
FRAME_STACK_K = 4  # Must match training


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def load_net(weight_file: Path, n_actions: int, obs_shape) -> DQN:
    """Load a trained DQN from weights file."""
    net = DQN(obs_shape, n_actions, frame_stack=FRAME_STACK_K).to(DEVICE)
    net.load_state_dict(torch.load(weight_file, map_location=DEVICE, weights_only=True))
    net.eval()
    return net


def is_display_available() -> bool:
    """Check if display is available for OpenCV."""
    if not CV2_AVAILABLE:
        return False
    
    # Only check DISPLAY for Linux (not macOS)
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    
    # Try to create a test window
    try:
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL PLAY LOOP (with OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════
def play_visual(
    layout: str,
    net: DQN,
    episodes: int,
    delay_ms: int,
    scale: int,
    show_qvalues: bool = True
):
    """Play with OpenCV visualization."""
    env = PacmanEnv(layout, reward_shaping=False)
    frame_stack = FrameStack(k=FRAME_STACK_K)
    
    wins = 0
    total_steps = []
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    cv2.namedWindow("Pac-Man DQN", cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)  # Let window initialize

    for ep in range(1, episodes + 1):
        raw_state, _ = env.reset()
        state = frame_stack.reset(raw_state)
        
        done = False
        step = 0
        win = False
        ep_reward = 0

        while not done and step < 1000:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                q_values = net(state_tensor)
                action = int(q_values.argmax())
                q_vals = q_values.squeeze().cpu().numpy()

            raw_next_state, reward, done, _, _ = env.step(action)
            state = frame_stack.step(raw_next_state)
            step += 1
            ep_reward += reward
            
            if done and not env.pellets:
                win = True

            # Render frame
            frame = cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR)
            
            # Scale up
            if scale > 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)

            # Add HUD
            h, w = frame.shape[:2]
            
            # Top bar: layout, episode, step
            cv2.putText(frame, f"{layout} | Ep {ep}/{episodes} | Step {step}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Q-values display
            if show_qvalues:
                y_offset = h - 60
                cv2.putText(frame, "Q-values:", (5, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                for i, (name, q) in enumerate(zip(action_names, q_vals)):
                    color = (0, 255, 0) if i == action else (150, 150, 150)
                    text = f"{name}: {q:.2f}"
                    cv2.putText(frame, text, (5 + (i % 2) * 80, y_offset + 15 + (i // 2) * 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            # Pellets remaining
            cv2.putText(frame, f"Pellets: {len(env.pellets)}",
                        (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imshow("Pac-Man DQN", frame)
            
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                print("\nQuit requested.")
                env.close()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord(' '):  # Pause
                cv2.waitKey(0)
            elif key == ord('+') or key == ord('='):  # Speed up
                delay_ms = max(10, delay_ms - 20)
            elif key == ord('-'):  # Slow down
                delay_ms = min(500, delay_ms + 20)

        wins += win
        total_steps.append(step)
        result = "WIN" if win else "LOSE"
        print(f"Episode {ep:3d}: {result} in {step:4d} steps | Reward: {ep_reward:.1f}")

    # Final summary
    print(f"\n{'='*50}")
    print(f"RESULTS for {layout}:")
    print(f"  Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"  Avg steps: {np.mean(total_steps):.1f}")
    print(f"{'='*50}")
    
    env.close()
    cv2.destroyAllWindows()
    
    return wins / episodes


# ═══════════════════════════════════════════════════════════════════════════════
# HEADLESS PLAY LOOP (text output only)
# ═══════════════════════════════════════════════════════════════════════════════
def play_headless(layout: str, net: DQN, episodes: int, verbose: bool = True):
    """Play without display - text output only."""
    env = PacmanEnv(layout, reward_shaping=False)
    frame_stack = FrameStack(k=FRAME_STACK_K)
    
    wins = 0
    total_steps = []
    total_rewards = []
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    print(f"\n{'='*50}")
    print(f"Running DQN agent on {layout} layout (headless)")
    print(f"Device: {DEVICE}")
    print(f"{'='*50}")

    for ep in range(1, episodes + 1):
        raw_state, _ = env.reset()
        state = frame_stack.reset(raw_state)
        
        done = False
        step = 0
        win = False
        ep_reward = 0
        
        initial_pellets = len(env.pellets)
        
        if verbose:
            print(f"\nEpisode {ep}/{episodes}: Starting")
            print(f"  Initial: Pac-Man at {env.pac_pos}, Ghost(s) at {env.ghost_pos}")

        while not done and step < 1000:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                q_values = net(state_tensor)
                action = int(q_values.argmax())

            # Verbose logging for first few steps
            if verbose and (step < 5 or step % 100 == 0):
                q_vals = q_values.squeeze().cpu().numpy()
                print(f"  Step {step:4d}: Action={action_names[action]:5s}, "
                      f"Q={[f'{q:.2f}' for q in q_vals]}, "
                      f"Pellets={len(env.pellets)}")

            raw_next_state, reward, done, _, _ = env.step(action)
            state = frame_stack.step(raw_next_state)
            step += 1
            ep_reward += reward

            if done and not env.pellets:
                win = True

        wins += win
        total_steps.append(step)
        total_rewards.append(ep_reward)
        
        pellets_collected = initial_pellets - len(env.pellets)
        result = "WIN" if win else "LOSE"
        
        print(f"Episode {ep:3d}: {result} | Steps: {step:4d} | "
              f"Pellets: {pellets_collected}/{initial_pellets} | "
              f"Reward: {ep_reward:.1f}")

    # Final summary
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS for {layout}:")
    print(f"  Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"  Avg steps: {np.mean(total_steps):.1f}")
    print(f"  Avg reward: {np.mean(total_rewards):.1f}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*50}")

    env.close()
    
    return wins / episodes


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE ALL LAYOUTS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_all(model_path: Path, episodes: int = 50) -> dict[str, float]:
    """Evaluate model on all layouts and compute competition score."""
    layouts = ["classic", "spiral", "spiral_harder", "empty"]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION: {model_path}")
    print(f"Episodes per layout: {episodes}")
    print(f"{'='*60}")

    for layout in layouts:
        print(f"\n>>> Evaluating on {layout}...")
        
        env = PacmanEnv(layout, reward_shaping=False)
        n_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        
        net = load_net(model_path, n_actions, obs_shape)
        win_rate = play_headless(layout, net, episodes, verbose=False)
        results[layout] = win_rate * 100
        
        env.close()

    # Competition score (from requirements PDF)
    avg_score = np.mean(list(results.values()))
    
    print(f"\n{'='*60}")
    print("COMPETITION SCORE:")
    print(f"{'='*60}")
    for layout, win_rate in results.items():
        print(f"  {layout:15s}: {win_rate:.1f}%")
    print(f"  {'-'*25}")
    print(f"  {'AVERAGE':15s}: {avg_score:.1f}%")
    print(f"{'='*60}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PLAY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def play(
    layout: str,
    net: DQN,
    episodes: int,
    delay_ms: int = 100,
    scale: int = 3,
    headless: bool = False
):
    """Play with automatic display detection or explicit headless mode."""
    if headless or not is_display_available():
        if not headless:
            print("No display available - running in headless mode")
        return play_headless(layout, net, episodes)
    else:
        return play_visual(layout, net, episodes, delay_ms, scale)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play/evaluate trained Pac-Man DQN")
    
    parser.add_argument(
        "--layout",
        choices=["classic", "empty", "spiral", "spiral_harder"],
        default="classic",
        help="Layout to play"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to .pt weight file (default: pacman_dqn_<layout>.pt)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to play"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        help="Delay between frames in ms (visual mode only)"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Integer upscale factor for window size"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (text output only)"
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate on all layouts and compute competition score"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed step-by-step output in headless mode"
    )

    args = parser.parse_args()

    # Full evaluation mode
    if args.eval_all:
        model_path = args.model or Path("pacman_dqn_multitask.pt")
        if not model_path.exists():
            # Try layout-specific
            model_path = Path(f"pacman_dqn_{args.layout}.pt")
        
        if not model_path.exists():
            sys.exit(f"Error: weight file {model_path} not found")
        
        evaluate_all(model_path, args.episodes)
        sys.exit(0)

    # Single layout mode
    weight_path = args.model or Path(f"pacman_dqn_{args.layout}.pt")
    
    # Try multitask model if layout-specific not found
    if not weight_path.exists():
        multitask_path = Path("pacman_dqn_multitask.pt")
        if multitask_path.exists():
            print(f"Layout-specific weights not found, using {multitask_path}")
            weight_path = multitask_path
        else:
            sys.exit(f"Error: weight file {weight_path} not found")

    # Build temp env to get shapes
    tmp_env = PacmanEnv(args.layout)
    obs_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    # Load network and play
    net = load_net(weight_path, n_actions, obs_shape)
    print(f"Loaded model from {weight_path}")
    print(f"Using device: {DEVICE}")
    
    play(args.layout, net, args.episodes, args.speed, args.scale, args.headless)