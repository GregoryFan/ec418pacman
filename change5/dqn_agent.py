# dqn_agent.py - Fixed "Blindness" bug, kept Beefy CNN & NoisyNets

from __future__ import annotations
import math, random
from typing import Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Noisy Linear Layer (Unchanged)
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# Dueling DQN with Beefy CNN and NoisyNet
class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        """
        obs_shape: (H, W, C)
        """
        super().__init__()

        if len(obs_shape) != 3:
            raise ValueError(f"Expected obs_shape (H, W, C), got {obs_shape}")
        
        h, w, c = obs_shape
        in_channels = c 

        # Beefy CNN Architecture
        # Removed padding to keep dimensions tighter and more standard for RL
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Beefy layer: 128 filters instead of standard 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamic calculation of flattened size
        # This fixes the "blindness" by ensuring we flatten correctly 
        # instead of pooling everything to 1x1
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, h, w)
            dummy_output = self.features(dummy_input)
            self.feature_dim = dummy_output.view(1, -1).size(1)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, 1),
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions),
        )

    def reset_noise(self):
        """Resample noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # x: (H, W, C) or (B, H, W, C)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        feats = self.features(x)
        
        # CRITICAL FIX: Flatten the features, DO NOT POOL
        flattened = feats.reshape(x.size(0), -1)

        V = self.value_stream(flattened)
        A = self.adv_stream(flattened)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


# Prioritized Experience Replay (Unchanged)
class ReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, discount):
        # Force uint8 to prevent memory explosion
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)
        
        max_prio = max(self.priorities, default=1.0)
        transition = (state, action, reward, next_state, done, discount)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones, discounts = zip(*samples)

        states      = np.stack(states)
        next_states = np.stack(next_states)
        actions     = np.array(actions, dtype=np.int64)
        rewards     = np.array(rewards, dtype=np.float32)
        dones       = np.array(dones, dtype=np.float32)
        discounts   = np.array(discounts, dtype=np.float32)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, discounts, indices, weights

    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)


# Action Selection (Unchanged)
def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int,
                  n_actions: int) -> int:
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
    net.reset_noise()

    if random.random() < eps:
        return random.randrange(n_actions)

    with torch.no_grad():
        q = net(torch.as_tensor(state, device=DEVICE))
        return int(q.argmax())


# Optimisation Loop (Unchanged)
def optimise(memory: ReplayMemory,
             policy: DQN,
             target: DQN,
             optimiser: optim.Optimizer,
             batch: int,
             step: int,
             beta_start: float = 0.4,
             beta_frames: int = 100_000):
    
    if len(memory) < batch:
        return

    beta = min(1.0, beta_start + (1.0 - beta_start) * step / beta_frames)
    s, a, r, s2, d, discounts, indices, weights = memory.sample(batch, beta)

    states      = torch.as_tensor(s,    device=DEVICE, dtype=torch.float32)
    next_states = torch.as_tensor(s2,   device=DEVICE, dtype=torch.float32)
    actions     = torch.as_tensor(a,    device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards     = torch.as_tensor(r,    device=DEVICE, dtype=torch.float32)
    dones       = torch.as_tensor(d,    device=DEVICE, dtype=torch.float32)
    discounts   = torch.as_tensor(discounts, device=DEVICE, dtype=torch.float32)
    weights_t   = torch.as_tensor(weights,   device=DEVICE, dtype=torch.float32)

    q_values = policy(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_online = policy(next_states)
        next_actions  = next_q_online.argmax(dim=1)
        next_q_target = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + (1.0 - dones) * discounts * next_q_target

    td_errors = q_values - target_q
    per_loss = (F.smooth_l1_loss(q_values, target_q, reduction='none') * weights_t).mean()

    optimiser.zero_grad()
    per_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()

    new_prios = td_errors.detach().abs().cpu().numpy() + 1e-6
    memory.update_priorities(indices, new_prios)