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
#from dqn_agent import DuelingDQN, ReplayMemory, select_action, optimise, DEVICE
#from frame_stack import FrameStack
# new
from dqn_noisy import DQN, PrioritizedReplayMemory, ReplayMemory, select_action, optimise, DEVICE

# ───────── hyper‑parameters ─────────
NUM_EPISODES      = 1000  # Increased for better convergence, especially classic
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 200
BATCH_SIZE        = 128
MEMORY_CAP        = 50_000  # Larger buffer for classic layout
MEMORY_CAP_SMALL  = 20_000  # For smaller layouts
GAMMA             = 0.99
LR                = 1e-3
LR_DECAY          = 0.995  # Learning rate decay per episode
LR_MIN            = 1e-5   # Minimum learning rate
USE_PRIORITIZED   = True   # Use Prioritized Experience Replay
EPS               = (1.0, 0.05, 8000)   # ε‑greedy schedule (start, end, decay) - not used with noisy nets


# ───────── single‑layout trainer ─────────
def train_layout(layout: str, episodes: int) -> Path:
    env = PacmanEnv(layout)
    obs_shape = env.observation_space.shape        # (H, W, C)
    n_actions = env.action_space.n
    
    #updated fro framestacking
    #env = PacmanEnv(layout)
    #env = FrameStack(env, k=4)          # stack 4 frames
    #obs_shape = env.shape                # now (H, W, C*4)
    #n_actions = env.env.action_space.n   # note the extra .env
    
    print("Created environment")
    # Use deeper network for classic layout due to complexity
    use_deeper = layout == "classic"
    policy  = DQN(obs_shape, n_actions, deeper=use_deeper).to(DEVICE)
    target  = DQN(obs_shape, n_actions, deeper=use_deeper).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    print("Created policy and target networks")
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    
    # Use larger buffer for classic layout, prioritized replay if enabled
    memory_cap = MEMORY_CAP if layout == "classic" else MEMORY_CAP_SMALL
    if USE_PRIORITIZED:
        memory = PrioritizedReplayMemory(memory_cap, alpha=0.6, beta=0.4, beta_increment=1e-6)
        print(f"Created PrioritizedReplayMemory with capacity {memory_cap}")
    else:
        memory = ReplayMemory(memory_cap)
        print(f"Created ReplayMemory with capacity {memory_cap}")
    
    print("Created optimizer and memory")
    step = 0
    current_lr = LR
    WARMUP_STEPS = 1000  # Collect experiences before training starts
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, ep_reward = False, 0.0
        print(f"[{layout}, episode {ep}].")
        max_steps = 4000
        steps_in_ep = 0
        while not done: #and steps_in_ep < max_steps:
            # Reset noise for exploration
            ###IMPORTANT TO KEEP IT INCREASES SCORE FOR SPIRAL
            policy.reset_noise()
            target.reset_noise()

            #action = select_action(state, policy, step, *EPS)
            action = select_action(state, policy) #for the noise

            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            # Train more frequently (every step) but only after warmup
            if step >= WARMUP_STEPS:
                optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0 and step > 0:
                target.load_state_dict(policy.state_dict())

            #steps_in_ep += 1
            step += 1

        # Learning rate decay
        current_lr = max(LR_MIN, current_lr * LR_DECAY)
        for param_group in optimiser.param_groups:
            param_group['lr'] = current_lr
        
        if ep % 50 == 0 or ep == episodes:  # More frequent logging
            print(f"[{layout}] Episode {ep:4d} | reward = {ep_reward:6.1f} | lr = {current_lr:.6f} | memory = {len(memory)} | steps = {step}")

    env.close()
 
    weight_path = Path(f"pacman_dqn_{layout}.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"[{layout}] training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on all Pac‑Man layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200‑episode run per layout instead of full 4000"
    )
    args = parser.parse_args()
    episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES


    for layout in ["classic"]:
    #for layout in ["classic", "spiral", "spiral_harder", "empty"]:  
        train_layout(layout, episodes)

