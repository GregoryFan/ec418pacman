#!/usr/bin/env python
"""
train.py - Speedrun Optimized
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.optim as optim
import gc 

from pacman_env import PacmanEnv
from dqn_agent import DQN, ReplayMemory, select_action, optimise, DEVICE

# hyper-parameters 
NUM_EPISODES_BASE = 1000      
NUM_EPISODES_FAST = 10      
TARGET_FREQ       = 500
BATCH_SIZE        = 64
MEMORY_CAP        = 20_000    
GAMMA             = 0.99

LR_BASE           = 1e-3      
LR_FINETUNE       = 5e-4      

# Epsilon schedules 
EPS               = (1.0, 0.05, 8_000)    
FINE_TUNE_EPS     = (0.3, 0.05, 20_000)   

N_STEPS = 3
STACK_SIZE = 4 

def current_eps(step: int, eps_schedule) -> float:
    eps_start, eps_end, eps_decay = eps_schedule
    return eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)

def to_gray(obs_rgb: np.ndarray) -> np.ndarray:
    # 1. Boost Pellet Visibility (Optional but recommended for Classic)
    # Detect Green-ish pixels (Pellets) and turn them Pure White
    # This prevents them from fading out during resizing
    mask = (obs_rgb[:, :, 1] > 150) & (obs_rgb[:, :, 0] < 100) & (obs_rgb[:, :, 2] < 100)
    obs_rgb[mask] = [255, 255, 255]

    # 2. Standard Grayscale Conversion
    gray = np.dot(obs_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # 3. Resize to standard 84x84 (Preserves Transfer Learning compatibility)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    return gray.astype(np.uint8)

def make_initial_stack(frame: np.ndarray, stack_size: int = STACK_SIZE) -> np.ndarray:
    return np.stack([frame] * stack_size, axis=-1)

def update_stack(stack: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
    return np.concatenate([stack[..., 1:], new_frame[..., None]], axis=-1)

def train_layout(
    layout: str,
    episodes: int,
    init_weights: Path | None = None,
    eps_schedule = EPS,
    lr: float = LR_BASE,
    memory_cap: int = MEMORY_CAP,
) -> Path:
    """
    Train a DQN on a single layout and return the path to the saved weights.
    """
    env = PacmanEnv(layout) # Ensure you use the updated pacman_env.py with the rewards fixed!

    state_rgb, _ = env.reset()
    frame = to_gray(state_rgb)              
    stack = make_initial_stack(frame)       
    obs_shape = stack.shape                 
    n_actions = env.action_space.n

    policy  = DQN(obs_shape, n_actions).to(DEVICE)
    target  = DQN(obs_shape, n_actions).to(DEVICE)

    if init_weights is not None:
        state_dict = torch.load(init_weights, map_location=DEVICE)
        policy.load_state_dict(state_dict)
        target.load_state_dict(state_dict)
        print(f"[{layout}] Loaded initial weights from {init_weights}")
    else:
        target.load_state_dict(policy.state_dict())
        print(f"[{layout}] Initialised policy and target networks from scratch")

    optimiser = optim.Adam(policy.parameters(), lr=lr)
    memory    = ReplayMemory(memory_cap)

    print(f"[{layout}] Training on device: {DEVICE} for {episodes} episodes")

    ep_rewards: list[float] = []
    ep_lengths: list[int]   = []
    ep_wins:    list[int]   = []

    # --- INSERT THIS LINE HERE ---
    best_avg_reward = -float('inf') 
    # -----------------------------

    step = 0  

    for ep in range(1, episodes + 1):
        state_rgb, _ = env.reset()
        frame = to_gray(state_rgb)
        state = make_initial_stack(frame)   

        done = False
        ep_reward = 0.0
        ep_len = 0

        n_step_buffer: deque = deque(maxlen=N_STEPS)

        while not done and ep_len < 2000:
            action = select_action(
                state, policy, step,
                eps_schedule[0], eps_schedule[1], eps_schedule[2],
                n_actions,
            )
            step += 1
            ep_len += 1

            next_rgb, reward, done, _, _ = env.step(action)
            next_frame = to_gray(next_rgb)
            next_state = update_stack(state, next_frame)

            n_step_buffer.append((state, action, reward, next_state, float(done)))

            if len(n_step_buffer) == N_STEPS:
                R = 0.0
                k = 0
                for i, (_, _, r_i, _, d_i) in enumerate(n_step_buffer):
                    R += (GAMMA ** i) * r_i
                    k += 1
                    if d_i: break

                s0, a0, _, _, _ = n_step_buffer[0]
                s_n = n_step_buffer[k - 1][3]   
                d_n = n_step_buffer[k - 1][4]   
                discount_factor = GAMMA ** k

                memory.push(s0, a0, R, s_n, d_n, discount_factor)

            state = next_state
            ep_reward += reward

            optimise(memory, policy, target, optimiser, BATCH_SIZE, step)


            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        # Flush buffer
        while len(n_step_buffer) > 0:
            R = 0.0
            k = 0
            for i, (_, _, r_i, _, d_i) in enumerate(n_step_buffer):
                R += (GAMMA ** i) * r_i
                k += 1
                if d_i: break
            s0, a0, _, _, _ = n_step_buffer[0]
            s_n = n_step_buffer[k - 1][3]
            d_n = n_step_buffer[k - 1][4]
            discount_factor = GAMMA ** k
            memory.push(s0, a0, R, s_n, d_n, discount_factor)
            n_step_buffer.popleft()

        win = 1 if (done and not env.pellets) else 0
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        ep_wins.append(win)
        
        # Clear memory
        if ep % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Logging
        if ep % 50 == 0 or ep == episodes:
            window = min(100, len(ep_rewards))
            avg_reward = np.mean(ep_rewards[-window:])
            avg_win    = np.mean(ep_wins[-window:]) * 100.0
            print(f"[{layout}] Ep {ep:4d} | R={ep_reward:6.1f} | win={win} | avg_R={avg_reward:6.1f} | avg_win={avg_win:4.1f}%")

            # --- INSERT THIS CODE ---
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = Path(f"pacman_dqn_{layout}_BEST.pt")
                torch.save(policy.state_dict(), best_path)
                print(f"[{layout}] NEW BEST FOUND! Saved to {best_path}")
                
            # If you specifically want to save the first model that wins:
            if win == 1 and not Path(f"pacman_dqn_{layout}_FIRST_WIN.pt").exists():
                 torch.save(policy.state_dict(), Path(f"pacman_dqn_{layout}_FIRST_WIN.pt"))
                 print(f"[{layout}] FIRST WIN! Saved special checkpoint.")
            # ------------------------

        # --- SAFEGUARD CHECKPOINT ---
        # Save every 500 episodes in case the SCC job dies early
        if ep % 500 == 0:
            ckpt_path = Path(f"pacman_dqn_{layout}_checkpoint.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[{layout}] SAFEGUARD: Saved checkpoint to {ckpt_path}")

    env.close()
    weight_path = Path(f"pacman_dqn_{layout}.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[{layout}] training finished  {weight_path.resolve()}")
    
    # Save CSV logs
    log_path = Path(f"train_log_{layout}.csv")
    data = np.column_stack([np.arange(1, episodes + 1), np.array(ep_rewards), np.array(ep_lengths), np.array(ep_wins)])
    np.savetxt(log_path, data, fmt=["%d", "%.4f", "%d", "%d"], delimiter=",", header="episode,reward,length,win")
    
    return weight_path


# 3-HOUR SPEEDRUN SCHEDULE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classic Only Run")
    parser.add_argument("--fast", action="store_true", help="Debug mode")
    args = parser.parse_args()

    # Determine episodes based on debug flag
    episodes = NUM_EPISODES_FAST if args.fast else 12_000 
    
    print(f">>> STARTING DIRECT TRAINING ON CLASSIC ({episodes} episodes)")
    
    # We use the BASE settings because we are starting from scratch
    w_classic = train_layout(
        "classic",
        episodes=episodes,
        init_weights=None,      # <--- CRITICAL: Start from scratch
        eps_schedule=EPS,       # <--- CRITICAL: Start exploring at 1.0 (100%), not 0.3
        lr=LR_BASE,             # <--- CRITICAL: Use the higher learning rate (1e-3)
        # Ensure you fixed the 'optimise' bug in train_layout discussed previously!
    )
