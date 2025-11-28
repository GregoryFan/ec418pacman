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
from dqn_agent import DQN, ReplayMemory, select_action, optimise, DEVICE
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

def stack_frames(buffer, new_frame):
    buffer.append(new_frame)
    while len(buffer) < 4:
        buffer.append(new_frame)
    return np.concatenate(list(buffer), axis=2)   # (84,84,12)

# ───────── hyper‑parameters ─────────
LAYOUTS = ["classic", "empty", "spiral", "spiral_harder"]

NUM_EPISODES      = 1000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 200
BATCH_SIZE        = 128
MEMORY_CAP        = 20_000
GAMMA             = 0.99
LR                = 1e-3
EPS               = (1.0, 0.05, 8_000)   # ε‑greedy schedule (start, end, decay)

# ───────── multi‑layout trainer ─────────
def train_multi_layout(episodes: int) -> Path:
    obs_shape = (84, 84, 12)
    n_actions = PacmanEnv("empty").action_space.n

    # Initialize dueling DQN
    policy  = DQN(obs_shape, n_actions).to(DEVICE)
    target  = DQN(obs_shape, n_actions).to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimiser = optim.Adam(policy.parameters(), lr=LR, weight_decay=1e-5)
    memories    = {layout: ReplayMemory(MEMORY_CAP // len(LAYOUTS)) 
                   for layout in LAYOUTS}

    step = 0
    for ep in range(1, episodes + 1):
        layout = random.choices(
            LAYOUTS,
            weights=[2.0, 2.0, 1.0, 1.0]  # classic y empty with more prob to be elected
        )[0] 
        env = PacmanEnv(layout)
        state_raw, _ = env.reset() # (H, W, 3)
        f = preprocess(state_raw)
        buffer = init_buffer()
        state = stack_frames(buffer, f)

        done, ep_reward = False, 0.0
        print(f"[{layout}, episode {ep}].")
        while not done:
            action = select_action(state, policy, step, *EPS)
            step += 1

            next_state_raw, reward, done, _, _ = env.step(action)
            f2 = preprocess(next_state_raw)
            next_state = stack_frames(buffer, f2)
            
            memories[layout].push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            layout_for_update = random.choice(LAYOUTS)
            optimise(memories[layout_for_update], policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())
                print(f"[SYNC] Synchronized target network in step {step}")
        
        env.close()
        if ep % 100 == 0 or ep == episodes:
            print(f"[{layout}] Episode {ep:4d} | reward = {ep_reward:6.1f}")
        
    weight_path = Path(f"pacman_dqn_dueling_multi_task_resnet.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[multi-DQN] training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SINGLE DQN over ALL layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200-episode run instead of full 1000"
    )
    args = parser.parse_args()
    episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES

    train_multi_layout(episodes)
