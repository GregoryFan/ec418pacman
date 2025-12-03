#!/usr/bin/env python
"""
train.py   train a separate DQN for every board layout
           ("classic", "empty", "spiral", "spiral_harder").

Saved weight files:
    pacman_dqn_<layout>.pt
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

from pacman_env import PacmanEnv
from dqn_agent import DQN, ReplayMemory, select_action, optimise, DEVICE

# hyper-parameters 
NUM_EPISODES_BASE = 1000      # base episodes for "normal" maps
NUM_EPISODES_FAST = 10      # for --fast debugging
TARGET_FREQ       = 200
BATCH_SIZE        = 128
MEMORY_CAP        = 50_000    # larger replay buffer
GAMMA             = 0.99

LR_BASE           = 1e-3      # for training from scratch
LR_FINETUNE       = 5e-4      # for harder / longer fine-tunes

# Epsilon schedules (you can later set these to (0,0,1) to rely fully on NoisyNets)
EPS               = (1.0, 0.05, 8_000)    
FINE_TUNE_EPS     = (0.3, 0.05, 20_000)   

# Multi-step returns
N_STEPS = 3

# Frame stacking
STACK_SIZE = 4  # number of frames in the stack; must match play_cv.py & dqn_agent usage


# helper for epsilon logging 
def current_eps(step: int, eps_schedule) -> float:
    """Compute epsilon given a global step and (start, end, decay)."""
    eps_start, eps_end, eps_decay = eps_schedule
    return eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)


#  preprocessing: grayscale + frame stacking
def to_gray(obs_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB observation (H, W, 3) to 84x84 uint8 grayscale.
    """
    gray = np.dot(obs_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return gray.astype(np.uint8)


def make_initial_stack(frame: np.ndarray, stack_size: int = STACK_SIZE) -> np.ndarray:
    """
    Given a single grayscale frame (H, W), return (H, W, stack_size)
    by repeating it across the channel axis.
    """
    return np.stack([frame] * stack_size, axis=-1)


def update_stack(stack: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
    """
    Given stack (H, W, stack_size) and new_frame (H, W),
    return updated stack with oldest frame dropped and new_frame appended.
    """
    return np.concatenate([stack[..., 1:], new_frame[..., None]], axis=-1)


# single-layout trainer
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

    env = PacmanEnv(layout)

    # Build a sample preprocessed state to infer obs_shape
    state_rgb, _ = env.reset()
    frame = to_gray(state_rgb)              # (84,84)
    stack = make_initial_stack(frame)       # (84,84,STACK_SIZE)
    obs_shape = stack.shape                 # (H, W, C)
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

    print(f"[{layout}] Created optimizer (lr={lr}) and memory (cap={memory_cap})")
    print(f"[{layout}] Obs shape = {obs_shape}, n_actions = {n_actions}")
    print(f"[{layout}] Training on device: {DEVICE}")

    # ---- logging buffers ----
    ep_rewards: list[float] = []
    ep_lengths: list[int]   = []
    ep_wins:    list[int]   = []

    step = 0  # global training step counter

    for ep in range(1, episodes + 1):
        state_rgb, _ = env.reset()
        frame = to_gray(state_rgb)
        state = make_initial_stack(frame)   # (H, W, STACK_SIZE)

        done = False
        ep_reward = 0.0
        ep_len = 0

        # N-step buffer: each entry is (state, action, reward, next_state, done)
        n_step_buffer: deque = deque(maxlen=N_STEPS)

        print(f"[{layout}, episode {ep}]")

        while not done:
            # epsilon (for logging / exploration; can later set EPS to zero if desired)
            eps_val = current_eps(step, eps_schedule)

            action = select_action(
                state,
                policy,
                step,
                eps_schedule[0],
                eps_schedule[1],
                eps_schedule[2],
                n_actions,
            )
            step += 1
            ep_len += 1

            next_rgb, reward, done, _, _ = env.step(action)
            next_frame = to_gray(next_rgb)
            next_state = update_stack(state, next_frame)

            # Push into N-step buffer
            n_step_buffer.append((state, action, reward, next_state, float(done)))

            # Once buffer is full, build an N-step transition from the oldest entry
            if len(n_step_buffer) == N_STEPS:
                R = 0.0
                k = 0
                for i, (_, _, r_i, _, d_i) in enumerate(n_step_buffer):
                    R += (GAMMA ** i) * r_i
                    k += 1
                    if d_i:
                        break

                s0, a0, _, _, _ = n_step_buffer[0]
                s_n = n_step_buffer[k - 1][3]   # next_state after k steps
                d_n = n_step_buffer[k - 1][4]   # done flag at that time
                discount_factor = GAMMA ** k

                memory.push(s0, a0, R, s_n, d_n, discount_factor)

            state = next_state
            ep_reward += reward

            # Optimise policy network with PER + N-step targets
            optimise(memory, policy, target, optimiser, BATCH_SIZE, step)

            # Periodically update target network
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        # Flush remaining N-step transitions after episode ends
        while len(n_step_buffer) > 0:
            R = 0.0
            k = 0
            for i, (_, _, r_i, _, d_i) in enumerate(n_step_buffer):
                R += (GAMMA ** i) * r_i
                k += 1
                if d_i:
                    break

            s0, a0, _, _, _ = n_step_buffer[0]
            s_n = n_step_buffer[k - 1][3]
            d_n = n_step_buffer[k - 1][4]
            discount_factor = GAMMA ** k

            memory.push(s0, a0, R, s_n, d_n, discount_factor)
            n_step_buffer.popleft()

        # episode finished: detect win (no pellets left)
        win = 1 if (done and not env.pellets) else 0

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        ep_wins.append(win)

        # Print per-episode and moving averages every 50 episodes
        if ep % 50 == 0 or ep == episodes:
            window = min(100, len(ep_rewards))
            avg_reward = np.mean(ep_rewards[-window:])
            avg_len    = np.mean(ep_lengths[-window:])
            avg_win    = np.mean(ep_wins[-window:]) * 100.0

            print(f"[{layout}] Ep {ep:4d} | "
                  f"R={ep_reward:6.1f} | "
                  f"len={ep_len:3d} | "
                  f"win={win} | "
                  f"avg_R(last {window})={avg_reward:6.1f} | "
                  f"avg_win={avg_win:4.1f}% "
                  f"| epsH{eps_val:.3f}")

    env.close()
    weight_path = Path(f"pacman_dqn_{layout}.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[{layout}] training finished  {weight_path.resolve()}")

    # save training log to CSV for this layout
    log_path = Path(f"train_log_{layout}.csv")
    data = np.column_stack([
        np.arange(1, episodes + 1),
        np.array(ep_rewards),
        np.array(ep_lengths),
        np.array(ep_wins),
    ])
    np.savetxt(
        log_path,
        data,
        fmt=["%d", "%.4f", "%d", "%d"],
        delimiter=",",
        header="episode,reward,length,win",
        comments="",
    )
    print(f"[{layout}] training log saved  {log_path.resolve()}")

    return weight_path


# CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on all Pac-Man layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200-episode run per layout instead of full 1000"
    )
    args = parser.parse_args()
    base_episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES_BASE

    # Phase 1: spiral
    w_spiral = train_layout(
        "spiral",
        base_episodes,
        eps_schedule=EPS,
        lr=LR_BASE,
    )

    # Phase 2: spiral_harder from scratch
    w_spiral_harder = train_layout(
        "spiral_harder",
        base_episodes,
        eps_schedule=EPS,
        lr=LR_BASE,
    )

    # Phase 3: empty, warm-started from spiral_harder
    w_empty = train_layout(
        "empty",
        base_episodes,
        init_weights=w_spiral_harder,
        eps_schedule=FINE_TUNE_EPS,
        lr=LR_FINETUNE,
    )

    # Phase 4: classic, also warm-started from spiral_harder
    w_classic = train_layout(
        "classic",
        base_episodes * 10,          # hardest layout gets more episodes
        init_weights=w_spiral_harder,
        eps_schedule=FINE_TUNE_EPS,
        lr=LR_FINETUNE,
    )
