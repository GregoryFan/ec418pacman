#!/usr/bin/env python
"""
train_dqn_multi.py – train the same dueling DQN for the 4 available board layouts
           ("classic", "empty", "spiral", "spiral_harder").

Saved weight files:
    pacman_dqn_<layout>.pt
"""

from __future__ import annotations
import argparse, torch, torch.optim as optim
from pathlib import Path
from pacman_env import PacmanEnv
from dqn_agent import DQN, PrioritizedReplayMemory, select_action, optimise, DEVICE
import random
import cv2
from collections import deque
import numpy as np

def preprocess(obs):
    # Resize to 84x84 (3 channels)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs

def init_buffer():
    return deque(maxlen=4)

def stack_frames(buffer: deque, new_frame: np.ndarray) -> np.ndarray:
    buffer.append(new_frame)
    while len(buffer) < 4:
        buffer.append(new_frame)
    return np.concatenate(list(buffer), axis=2)  # (84,84,12)

# ───────── hyper‑parameters ─────────
LAYOUTS = ["classic", "empty", "spiral", "spiral_harder"]

NUM_EPISODES      = 1000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 1000
BATCH_SIZE        = 128
MEMORY_CAP        = 50_000
GAMMA             = 0.99
LR                = 1e-4
EPS               = (1.0, 0.05, 20_000)   # ε‑greedy schedule (start, end, decay)

# ───────── simple curriculum ─────────
def choose_layout(ep: int) -> str:
    """
    1–300:    empty
    301–600:  classic
    601–800:  classic + empty + spiral
    801–1000: all
    """
    if ep <= 300:
        return "empty"
    elif ep <= 600:
        return "spiral"
    elif ep <= 800:
        return "spiral_harder"
    else:
        return "classic"

# ───────── multi‑layout trainer ─────────
def train_multi_layout(episodes: int) -> Path:
    obs_shape = (84, 84, 3)
    n_actions = PacmanEnv("empty").action_space.n

    # Initialize dueling DQN
    policy  = DQN(obs_shape, n_actions).to(DEVICE)
    target  = DQN(obs_shape, n_actions).to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory    = PrioritizedReplayMemory(MEMORY_CAP, alpha=0.6, beta_start=0.4, beta_frames=episodes * 200)

    step = 0
    for ep in range(1, episodes + 1):
        layout = choose_layout(ep)
        env = PacmanEnv(layout)
        
        state_raw, _ = env.reset() # (H, W, 3)
        state = preprocess(state_raw)

        done, ep_reward = False, 0.0
        print(f"[{layout}, episode {ep}].")
        while not done:
            action = select_action(state, policy, step, *EPS)
            step += 1

            next_state_raw, reward, done, _, _ = env.step(action)
            next_state = preprocess(next_state_raw)
            
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())
                print(f"[SYNC] Synchronized target network in step {step}")
        
        env.close()
        if ep % 50 == 0 or ep == episodes:
            print(f"[{layout}] Episode {ep:4d} | reward = {ep_reward:6.1f}")
        
    weight_path = Path(f"pacman_dqn_dueling_multi_task_per_curriculum.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[multi-DQN+PER] training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SINGLE Double DQN over ALL layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200-episode run instead of full 1000"
    )
    args = parser.parse_args()
    episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES

    train_multi_layout(episodes)
