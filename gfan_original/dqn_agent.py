# dqn_agent.py  –  network, replay buffer, ε‑greedy, optimiser
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use GPU if available (CUDA) or MPS (Apple GPU), else fallback to CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")   # fallback to CPU

print(f"Using device: {DEVICE}")
# ──────────────── network ────────────────
class DuelingDQN(nn.Module):
    """
    A simple Conv‐>FC Q‑network that adapts to any (H,W) image size.
    Pass the *observation‑space shape* when constructing.
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        H, W, C = obs_shape          # Gym shape order (H,W,C)
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy).shape[1]
        
        #self.fc = nn.Sequential(
         #   nn.Linear(conv_out, 512), nn.ReLU(),
         #   nn.Linear(512, n_actions),
        #)
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )


    def forward(self, x: torch.Tensor):
        # x arrives as (B,H,W,C) uint8 from env; convert to float & channels‑first
        x = x.float().permute(0,3,1,2) / 255.0
        #return self.fc(self.conv(x))

    ##added this change
        features = self.conv(x)
        value = self.value_stream(features)
        adv   = self.adv_stream(features)
        # Combine streams
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q

# ───────────── replay buffer ─────────────
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

# ───────────── ε‑greedy & optimise ───────
def select_action(state: np.ndarray, net: DuelingDQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int) -> int:
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
    if random.random() < eps:
        # return random.randrange(net.fc[-1].out_features)
        return random.randrange(net.n_actions)
    with torch.no_grad():
        q = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
        return int(q.argmax())

def optimise(memory: ReplayMemory, policy: DuelingDQN, target: DuelingDQN,
             optimiser: optim.Optimizer, batch: int, gamma: float):
    if len(memory) < batch:
        return
    s, a, r, s2, d = memory.sample(batch)
    s  = torch.as_tensor(s,  device=DEVICE)
    s2 = torch.as_tensor(s2, device=DEVICE)
    a  = torch.as_tensor(a,  device=DEVICE, dtype=torch.int64).unsqueeze(1)
    r  = torch.as_tensor(r,  device=DEVICE, dtype=torch.float32)
    d  = torch.as_tensor(d,  device=DEVICE, dtype=torch.float32)

    q   = policy(s).gather(1, a).squeeze(1)
    with torch.no_grad():
        q2  = target(s2).max(1)[0]
        tgt = r + (1.0 - d) * gamma * q2

    loss = nn.functional.smooth_l1_loss(q, tgt)
    optimiser.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()



    
    
