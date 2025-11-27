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
from dqn_noisy import DQN, ReplayMemory, select_action, optimise, DEVICE

# ───────── hyper‑parameters ─────────
NUM_EPISODES      = 1000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 200
BATCH_SIZE        = 128
MEMORY_CAP        = 10_000
GAMMA             = 0.99
LR                = 1e-3
EPS               = (1.0, 0.05, 8000)   # ε‑greedy schedule (start, end, decay)

#since empyt layout is difficult, need to have a slower decay
#EPS = (1.0, 0.05, 20000)  # slower decay, explore longer


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

            #if step % 4 == 0:
            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

            #steps_in_ep += 1
            step += 1

        if ep % 100 == 0 or ep == episodes:
            print(f"[{layout}] Episode {ep:4d} | reward = {ep_reward:6.1f}")

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


    for layout in ["spiral_harder"]:
    #for layout in ["classic", "spiral", "spiral_harder", "empty"]:  
        train_layout(layout, episodes)

