# dqn_agent_noisy.py – Dueling DQN with NoisyLinear layers, replay buffer, optimizer
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")   # fallback to CPU

print(f"Using device: {DEVICE}")

# ───────────── Noisy Linear Layer ─────────────
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x: torch.Tensor):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

# ───────────── Dueling DQN with Noisy Layers ─────────────
class DQN(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        H, W, C = obs_shape
        self.n_actions = n_actions

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy).shape[1]

        # Value stream with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )

        # Advantage stream with noisy layers
        self.adv_stream = nn.Sequential(
            NoisyLinear(conv_out, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # x: (B,H,W,C) uint8 → float, channels-first
        x = x.float().permute(0,3,1,2) / 255.0
        features = self.conv(x)
        value = self.value_stream(features)
        adv = self.adv_stream(features)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        # Reset noise for all NoisyLinear layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# ───────────── Replay Memory ─────────────
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

# ───────────── Action Selection ─────────────
def select_action(state: np.ndarray, net: DQN) -> int:
    with torch.no_grad():
        q_values = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
        return int(q_values.argmax())
#def select_action(state: np.ndarray, net: DQN, step: int, eps_start: float, eps_end: float, eps_decay: int) -> int:
#  this was suppose to help when i switched to test classic but stll had 0% 
#   eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * step / eps_decay)
#    if random.random() < eps:
#        return random.randrange(net.n_actions)
#    else:
#       with torch.no_grad():
#            q_values = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
#            return int(q_values.argmax())


# ───────────── Optimization ─────────────
def optimise(memory: ReplayMemory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch: int, gamma: float):
    if len(memory) < batch:
        return
    s, a, r, s2, d = memory.sample(batch)
    s  = torch.as_tensor(s, device=DEVICE)
    s2 = torch.as_tensor(s2, device=DEVICE)
    a  = torch.as_tensor(a, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    r  = torch.as_tensor(r, device=DEVICE, dtype=torch.float32)
    d  = torch.as_tensor(d, device=DEVICE, dtype=torch.float32)

    q = policy(s).gather(1, a).squeeze(1)
    with torch.no_grad():
        #q2 = target(s2).max(1)[0]
        #tgt = r + (1.0 - d) * gamma * q2
    #replaced the above with the following for higher success
        # Double DQN: select action with policy, evaluate with target
        next_actions = policy(s2).argmax(dim=1, keepdim=True)   # shape (B,1)
        q2 = target(s2).gather(1, next_actions).squeeze(1)       # shape (B,)
        tgt = r + (1.0 - d) * gamma * q2

    loss = nn.functional.smooth_l1_loss(q, tgt)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()
    policy.reset_noise()
    target.reset_noise()
