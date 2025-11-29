#!/usr/bin/env python3
"""
evaluate_dqn.py – Evaluates a trained DQN on all 4 Pac-Man layouts.

It computes:
    win_rate for each layout
    final score (average)
according to the project evaluation methodology.
"""

import torch
import numpy as np
from pathlib import Path
from pacman_env import PacmanEnv
from dqn_agent import DQN, DEVICE
import cv2
from collections import deque


LAYOUTS = ["classic", "spiral", "spiral_harder", "empty"]
EPISODES_PER_LAYOUT = 50   # total 200 games
MODEL_PATH = "pacman_dqn_dueling_multi_task_per_curriculum.pt"

def preprocess(obs):
    return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

def greedy_action(net, state):
    """Select action = argmax Q(s,a)"""
    s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q = net(s)
        a = int(q.argmax(dim=1).item())
    return a


def evaluate_agent(model_path: str, episodes_per_layout: int = 50):
    results = {}

    for layout in LAYOUTS:
        env = PacmanEnv(layout)

        wins = 0
        for ep in range(episodes_per_layout):
            state_raw, _ = env.reset()
            state = preprocess(state_raw)
            done = False

            while not done:
                action = greedy_action(net, state)
                next_state_raw, reward, done, _, info = env.step(action)
                state = preprocess(next_state_raw)

            # Win detection: project uses positive reward as success
            if done and not env.pellets:
                wins += 1

        env.close()
        win_rate = wins / episodes_per_layout
        results[layout] = win_rate
        print(f"Layout {layout:14s} | Win rate = {win_rate:.3f}")

    # Final score
    final_score = np.mean(list(results.values()))
    print("\nFinal evaluation score =", final_score)
    return results, final_score


# ───────────── MAIN ─────────────
if __name__ == "__main__":
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    obs_shape = (84, 84, 3)
    n_actions = PacmanEnv("empty").action_space.n

    net = DQN(obs_shape, n_actions).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    evaluate_agent(MODEL_PATH, EPISODES_PER_LAYOUT)
