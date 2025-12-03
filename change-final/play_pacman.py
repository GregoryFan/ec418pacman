#!/usr/bin/env python
"""
play_pacman.py - Animate a trained DQN Pac-Man agent with OpenCV

This is an alternative visualization script. For full functionality,
use play_cv.py which has more features.

Example:
    python play_pacman.py --layout classic --episodes 20
    python play_pacman.py --layout spiral --speed 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from pacman_env import PacmanEnv
from dqn_agent import DQN, FrameStack, DEVICE

# Configuration - must match training
FRAME_STACK_K = 4


def load_policy(model_path: Path, num_actions: int, obs_shape) -> DQN:
    """Instantiate a DQN and load weights."""
    policy = DQN(obs_shape, num_actions, frame_stack=FRAME_STACK_K).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    policy.eval()
    return policy


def play(
    layout: str,
    policy: DQN,
    episodes: int = 10,
    frame_delay_ms: int = 100,
    scale: int = 3,
) -> None:
    """Run episodes on the chosen layout, animating with OpenCV."""
    env = PacmanEnv(layout=layout, reward_shaping=False)
    frame_stack = FrameStack(k=FRAME_STACK_K)
    
    wins = 0
    total_steps = []

    cv2.namedWindow("Pac-Man DQN (press q to quit)", cv2.WINDOW_AUTOSIZE)

    for ep in range(1, episodes + 1):
        raw_state, _ = env.reset()
        state = frame_stack.reset(raw_state)
        
        done = False
        steps = 0
        won = False
        ep_reward = 0

        while not done and steps < 1000:
            # Greedy action from policy
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                q_values = policy(state_tensor)
                action = int(q_values.argmax(1).item())

            # Environment step
            raw_next_state, reward, done, _, _ = env.step(action)
            state = frame_stack.step(raw_next_state)
            steps += 1
            ep_reward += reward
            
            if done and not env.pellets:
                won = True

            # Render frame
            frame_rgb = env.render(mode="rgb_array")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Scale up for visibility
            if scale > 1:
                frame_bgr = cv2.resize(
                    frame_bgr, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_NEAREST
                )

            # HUD text
            hud = f"Layout: {layout} | Ep {ep}/{episodes} | Step {steps} | Pellets: {len(env.pellets)}"
            cv2.putText(
                frame_bgr, hud,
                org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("Pac-Man DQN (press q to quit)", frame_bgr)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            
            if key == ord("q"):
                print("Quit requested - exiting.")
                env.close()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord(" "):  # Pause
                cv2.waitKey(0)

        wins += won
        total_steps.append(steps)
        result = "WIN" if won else "LOSE"
        print(f"Episode {ep:2d}: {result} in {steps} steps | Reward: {ep_reward:.1f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Finished {episodes} episodes on '{layout}':")
    print(f"  Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"  Avg steps: {np.mean(total_steps):.1f}")
    print(f"{'='*50}")

    env.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Play trained Pac-Man with animation")
    parser.add_argument(
        "--layout",
        choices=["classic", "empty", "spiral", "spiral_harder"],
        default="classic",
        help="Which board layout to use",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to .pt weight file (default: pacman_dqn_<layout>.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of games to play"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        metavar="MS",
        help="Delay between frames in milliseconds (lower = faster)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Integer upscale factor for window size",
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model or Path(f"pacman_dqn_{args.layout}.pt")
    
    # Try multitask model if layout-specific not found
    if not model_path.is_file():
        multitask_path = Path("pacman_dqn_multitask.pt")
        if multitask_path.is_file():
            print(f"Note: Using {multitask_path} (layout-specific weights not found)")
            model_path = multitask_path
        else:
            print(f"Error: weights not found at {model_path.resolve()}", file=sys.stderr)
            sys.exit(1)

    # Build temp env to get dimensions
    tmp_env = PacmanEnv(layout=args.layout)
    n_actions = tmp_env.action_space.n
    obs_shape = tmp_env.observation_space.shape
    tmp_env.close()

    # Load and play
    print(f"Loading model from {model_path}")
    print(f"Device: {DEVICE}")
    
    policy = load_policy(model_path, n_actions, obs_shape)
    play(
        layout=args.layout,
        policy=policy,
        episodes=args.episodes,
        frame_delay_ms=args.speed,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()