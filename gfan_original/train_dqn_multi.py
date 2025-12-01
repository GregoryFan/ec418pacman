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

PHASES = [
    ("spiral", ["spiral"], 250),
    ("spiral_harder", ["spiral_harder"], 250),
    ("classic", ["classic"], 300),
    ("mixed", ["classic", "spiral", "spiral_harder", "empty"], 200),
]

NUM_EPISODES_FAST_FACTOR = 0.2 

# ───────── multi‑layout trainer ─────────
def train_curriculum(fast: bool = False) -> Path:
    phases = []
    for name, layouts, full_eps in PHASES:
        eps = int(full_eps * NUM_EPISODES_FAST_FACTOR) if fast else full_eps
        eps = max(eps, 10)
        phases.append((name, layouts, eps))

    tmp_env = PacmanEnv("empty")
    obs_shape = (84, 84, 3)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    # Initialize dueling DQN
    policy  = DQN(obs_shape, n_actions).to(DEVICE)
    target  = DQN(obs_shape, n_actions).to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory    = PrioritizedReplayMemory(MEMORY_CAP, 
                                        alpha=0.6, 
                                        beta_start=0.4, 
                                        beta_frames=200_000)

    global_step = 0
    for phase_name, layouts, num_eps in phases:
        print(f"\n=== Phase: {phase_name}  |  layouts={layouts}  "
              f"| episodes={num_eps} ===")
        
        for ep in range(1, num_eps+1):
            layout = random.choice(layouts)
            env = PacmanEnv(layout)
            
            state_raw, _ = env.reset() # (H, W, 3)
            state = preprocess(state_raw)

            done, ep_reward = False, 0.0
            print(f"[{phase_name} | {layout} | episode {ep}/{num_eps}]")

            while not done:
                action = select_action(state, policy, global_step, *EPS)
                global_step += 1

                next_state_raw, reward, done, _, _ = env.step(action)
                next_state = preprocess(next_state_raw)
                
                memory.push(state, action, reward, next_state, float(done))
                state = next_state
                ep_reward += reward

                optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
                if global_step % TARGET_FREQ == 0:
                    target.load_state_dict(policy.state_dict())
                    print(f"[SYNC] step={global_step}")
            
            env.close()
            if ep % 20 == 0 or ep == num_eps:
                print(f"  Episode {ep:4d} | total reward = {ep_reward:7.1f}")

        phase_path = Path(f"pacman_dqn_phase_{phase_name}.pt")
        torch.save(policy.state_dict(), phase_path)
        print(f"[SAVE] Phase {phase_name} weights → {phase_path.resolve()}")
        
    weight_path = Path(f"pacman_dqn_curriculum.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[multi-DQN+PER] training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with curriculum + PER over Pac-Man layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="use fewer episodes per phase (for debugging)"
    )
    args = parser.parse_args()

    train_curriculum(args.fast)
