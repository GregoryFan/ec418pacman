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


# ───────── hyper‑parameters ─────────
NUM_EPISODES      = 1000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 200
BATCH_SIZE        = 128
MEMORY_CAP        = 20_000
GAMMA             = 0.99
LR                = 1e-3
EPS               = (1.0, 0.05, 8_000)   # ε‑greedy schedule (start, end, decay)

FINE_TUNE_EPS = (0.3, 0.05, 8_000)  # smaller exploration when using pretrained weights

# ───────── single‑layout trainer ─────────
def train_layout(
    layout: str,
    episodes: int,
    init_weights: Path | None = None,
    eps_schedule = EPS,
) -> Path:

    # Edited to allow retraining a trained model
    env = PacmanEnv(layout)
    raw_shape = env.observation_space.shape
    obs_shape = (raw_shape[2], raw_shape[0], raw_shape[1])
    n_actions = env.action_space.n

    policy  = DQN(obs_shape, n_actions).to(DEVICE)
    target  = DQN(obs_shape, n_actions).to(DEVICE)

    if init_weights is not None:
        state_dict = torch.load(init_weights, map_location=DEVICE)
        policy.load_state_dict(state_dict)
        target.load_state_dict(state_dict)
    else:
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
            # action = select_action(state, policy, step, *EPS)
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

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on all Pac‑Man layouts")
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200‑episode run per layout instead of full 4000"
    )
    args = parser.parse_args()
    episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES


    # To train all
    # for layout in ["classic", "spiral", "spiral_harder", "empty"]:  
    #     train_layout(layout, episodes)

    # For prototyping try a single one
    # for layout in ["empty"]:
    #     train_layout(layout, episodes)

    # Curriculum/phase training
    # Train on classic 2x or 3x more
    # Phase A: train spiral_harder from scratch (full exploration)
    w_spiral_harder = train_layout("spiral_harder", episodes, eps_schedule=EPS)

    # Phase B: transfer to empty (reduced exploration) with twice as long 
    # w_empty = train_layout(
    #     "empty",
    #     episodes * 2,
    #     init_weights=w_spiral_harder,
    #     eps_schedule=FINE_TUNE_EPS,
    # )

    # # Phase C: transfer to classic (more episodes + reduced exploration)
    w_classic = train_layout(
        "classic",
        episodes * 3,
        init_weights=w_spiral_harder,
        eps_schedule=FINE_TUNE_EPS,
    )