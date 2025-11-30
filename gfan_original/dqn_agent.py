# dqn_agent.py  –  network, replay buffer, ε‑greedy, optimiser
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────── network ────────────────
class DQN(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()

        H, W, C = obs_shape

        self.convChain = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),  # big stride
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out_size = self.convChain(dummy).shape[1]

        self.head = DuelingHead(conv_out_size, n_actions)
    
    def forward(self, x: torch.Tensor):
        x = x.float() / 255.0
        x = torch.permute(x, (0, 3, 1, 2))

        x = self.convChain(x)
        x = self.head(x)
        return x

class DuelingHead(nn.Module):
    def __init__(self, in_features: int, n_actions: int):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x: torch.Tensor):
        value = self.value(x)
        advantage = self.advantage(x)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals


# ───────────── replay buffer ─────────────
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)

# ───────────── ε‑greedy & optimise ───────
def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int) -> int:
    #Get epsilon
    if step < eps_decay:
        eps = eps_start - (step / eps_decay) * (eps_start - eps_end)
    else:
        eps = eps_end

    #Pick random with probability eps
    if random.random() < eps:
        return random.randrange(4)
    
    #Otherwise, pick the highest Q-Value
    stateTensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        qVals = net(stateTensor)

    return int(torch.argmax(qVals, dim = 1).item())

def optimise(memory: ReplayMemory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch_size: int, gamma: float):
    
    #Not Enough Memories
    if len(memory) < batch_size:
        return 

    #Get Values
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    dones   = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
    next_states = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        next_actions = policy(next_states).argmax(dim=1)
        next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)

    q_vals = policy(states).gather(1, actions).squeeze(1)
    loss = nn.functional.smooth_l1_loss(q_vals, target_q)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()



    
    
