# dqn_agent.py  –  network, replay buffer, ε‑greedy, optimiser

# ──────────────── network ────────────────
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network (CNN + dueling + Double-DQN support) 
class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()

        # We know env gives RGB images: (H, W, 3)
        # We'll just hardcode 3 input channels and not play games with shapes.
        in_channels = 3

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Global average pooling: output (batch, 64, 1, 1)  (batch, 64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = 64

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        # x: (N, H, W, C) from env; convert to (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # cast to float & scale (in case it's uint8)
        x = x.to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        feats = self.features(x)
        pooled = self.global_pool(feats).view(x.size(0), -1)

        V = self.value_stream(pooled)              # (N, 1)
        A = self.adv_stream(pooled)               # (N, n_actions)

        Q = V + (A - A.mean(dim=1, keepdim=True)) # (N, n_actions)
        return Q


# replay buffer
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        s = random.sample(self.buf, batch_size)
        return map(np.array, zip(*s))

    def __len__(self):
        return len(self.buf)


#  e-greedy & optimise
def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int,
                  n_actions: int) -> int:
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = net(torch.as_tensor(state, device=DEVICE))
        return int(q.argmax())


def optimise(memory: ReplayMemory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch: int, gamma: float):

    if len(memory) < batch:
        return

    # Sample batch
    s, a, r, s2, d = memory.sample(batch)

    # Convert to tensors
    states      = torch.as_tensor(s,  device=DEVICE)
    next_states = torch.as_tensor(s2, device=DEVICE)
    actions     = torch.as_tensor(a,  device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards     = torch.as_tensor(r,  device=DEVICE, dtype=torch.float32).unsqueeze(1)
    dones       = torch.as_tensor(d,  device=DEVICE, dtype=torch.float32).unsqueeze(1)

    # Current Q-values: Q(s,a)
    q_values = policy(states).gather(1, actions)     # (B,1)

    # -------- Double DQN target --------
    with torch.no_grad():
        # Action selection from policy net
        next_actions = policy(next_states).argmax(dim=1, keepdim=True)  # (B,1)

        # Action evaluation from target net
        next_q = target(next_states).gather(1, next_actions)            # (B,1)

        # Bellman target
        target_q = rewards + (1.0 - dones) * gamma * next_q

    loss = F.smooth_l1_loss(q_values, target_q)

    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()



