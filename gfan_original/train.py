#!/usr/bin/env python
"""
train.py – train a separate DQN for every available board layout
           ("classic", "empty", "spiral", "spiral_harder").

Saved weight files:
    pacman_dqn_<layout>.pt
"""

from __future__ import annotations
import argparse, torch, torch.optim as optim
from pathlib import Path
from pacman_env import PacmanEnv
from dqn_agent import DQN, ReplayMemory, select_action, optimise, DEVICE
from frame_mem import FrameStack
import random

# ───────── hyper‑parameters ─────────
NUM_EPISODES      = 40000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 5000
BATCH_SIZE        = 128
MEMORY_CAP        = 100_000
GAMMA             = 0.99
LR                = 5e-5
EPS               = (1.0, 0.1, 500_000)   # ε‑greedy schedule (start, end, decay)

# ───────── single‑layout trainer ─────────
def train_layout(layout: str, episodes: int) -> Path:
    env = PacmanEnv(layout)
    env = FrameStack(env, k=4)
    obs_shape = env.observation_space.shape        # (H, W, C)
    n_actions = env.action_space.n
    print("Created environment")
    policy  = DQN(obs_shape, n_actions).to(DEVICE)
    target  = DQN(obs_shape, n_actions).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    print("Created policy and target networks")
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory    = ReplayMemory(MEMORY_CAP)
    print("Created optimizer and memory")
    step = 0
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, ep_reward = False, 0.0
        print(f"[{layout}, episode {ep}].")
        while not done:
            action = select_action(state, policy, step, *EPS)
            step += 1

            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        if ep % 100 == 0 or ep == episodes:
            print(f"[{layout}] Episode {ep:4d} | reward = {ep_reward:6.1f}")

    env.close()
    weight_path = Path(f"pacman_dqn_{layout}.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[{layout}] training finished → {weight_path.resolve()}")
    return weight_path

def train_all_layouts(layouts: list[str], total_episodes: int) -> Path:

    # Use the first layout to determine the observation space shape
    first_env = PacmanEnv(layouts[0])
    first_env = FrameStack(first_env, k=4)
    obs_shape = first_env.observation_space.shape
    n_actions = first_env.action_space.n
    first_env.close() # Close temporary env

    # Create all environments and wrap them in FrameStack
    envs = {layout: FrameStack(PacmanEnv(layout), k=4) for layout in layouts}
    print("Created all environments.")
    
    # Single Policy and Target Network, Single Replay Memory
    policy = DQN(obs_shape, n_actions).to(DEVICE)
    target = DQN(obs_shape, n_actions).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    print("Created shared policy and target networks.")
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory    = ReplayMemory(MEMORY_CAP)
    print("Created shared optimizer and memory.")

    step = 0
    
    # 2. Main Training Loop
    for ep in range(1, total_episodes + 1):
        # Select a layout randomly for this episode to ensure balanced experience
        layout_name = random.choice(layouts)
        env = envs[layout_name]
        
        state, _ = env.reset()
        done, ep_reward = False, 0.0
        
        print(f"[Episode {ep:4d}/{total_episodes}, Layout: {layout_name}].")
        
        while not done:
            action = select_action(state, policy, step, *EPS)
            step += 1

            next_state, reward, done, _, _ = env.step(action)
            # All experiences go into the shared memory
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            # The shared policy is updated using samples from the shared memory
            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        if ep % 50 == 0 or ep == total_episodes:
            print(f"[{layout_name}] Episode {ep:4d} | Total steps: {step} | reward = {ep_reward:6.1f}")
            
    for env in envs.values():
        env.close()
        
    weight_path = Path("pacman_dqn_all_layouts.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[ALL] training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single DQN on all Pac‑Man layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200‑episode run per layout instead of full 1000"
    )
    args = parser.parse_args()
    
    # Total episodes for the single run
    total_episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES
    
    # List of all layouts
    all_layouts = ["classic", "spiral", "spiral_harder", "empty"]  
    
    train_all_layouts(all_layouts, total_episodes)
