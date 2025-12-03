# dqn_noisy.py – Dueling DQN with NoisyLinear layers, Prioritized Replay, N-step returns
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple, Optional
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
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, deeper: bool = False):
        super().__init__()
        H, W, C = obs_shape
        self.n_actions = n_actions

        # Convolutional feature extractor
        if deeper:
            # Deeper network for complex layouts like classic
            self.conv = nn.Sequential(
                nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1), nn.ReLU(),  # Extra layer
                nn.Conv2d(128, 128, 3, 1), nn.ReLU(),  # Extra layer
                nn.Flatten(),
            )
        else:
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
        if deeper:
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_out, 512),
                nn.ReLU(),
                NoisyLinear(512, 256),  # Extra layer
                nn.ReLU(),
                NoisyLinear(256, 1)
            )
        else:
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_out, 512),
                nn.ReLU(),
                NoisyLinear(512, 1)
            )

        # Advantage stream with noisy layers
        if deeper:
            self.adv_stream = nn.Sequential(
                NoisyLinear(conv_out, 512),
                nn.ReLU(),
                NoisyLinear(512, 256),  # Extra layer
                nn.ReLU(),
                NoisyLinear(256, n_actions)
            )
        else:
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

# ───────────── Prioritized Experience Replay ─────────────
class PrioritizedReplayMemory:
    """Prioritized Experience Replay using a sum tree for efficient sampling."""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent
        self.beta = beta    # importance sampling exponent
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        # Sum tree for efficient priority sampling
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        self.tree = np.zeros(2 * self.tree_size - 1)
        self.data = [None] * capacity
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        
    def _propagate(self, idx: int, change: float):
        """Update priority in tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index given priority value."""
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])
    
    def push(self, *transition):
        """Add transition with max priority."""
        idx = self.pos + self.tree_size - 1
        self.data[self.pos] = tuple(transition)
        self._update(idx, self.max_priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int):
        """Sample batch with priorities, return indices and importance weights."""
        indices = []
        priorities = []
        segment = self.tree[0] / batch_size
        
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self._retrieve(0, s)
            indices.append(idx)
            priorities.append(self.tree[idx])
        
        indices = np.array([idx - self.tree_size + 1 for idx in indices])
        priorities = np.array(priorities)
        
        # Importance sampling weights
        probs = priorities / self.tree[0]
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.data[idx] for idx in indices]
        transitions = map(np.array, zip(*batch))
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, priority in zip(indices, priorities):
            tree_idx = idx + self.tree_size - 1
            self._update(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size

# ───────────── Standard Replay Memory (for backward compatibility) ─────────────
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        s = random.sample(self.buf, batch_size)
        return map(np.array, zip(*s)), None, None

    def __len__(self):
        return len(self.buf)
    
    def update_priorities(self, *args):
        pass  # No-op for standard replay

# ───────────── Action Selection ─────────────
def select_action(state: np.ndarray, net: DQN, training: bool = True, epsilon: float = 0.0) -> int:
    """
    Select action with optional epsilon-greedy exploration and tie-breaking.
    
    Args:
        state: Current state
        net: DQN network
        training: If True, add small random noise to break ties
        epsilon: Probability of random action (for additional exploration)
    """
    # Small epsilon-greedy even with noisy networks to break out of stuck states
    if training and random.random() < epsilon:
        return random.randrange(net.n_actions)
    
    with torch.no_grad():
        q_values = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
        q_vals = q_values.squeeze(0).cpu().numpy()
        
        # If Q-values are too similar (within 0.1), add small noise to break ties
        if training and q_vals.max() - q_vals.min() < 0.1:
            # Add small random noise to break ties
            q_vals = q_vals + np.random.normal(0, 0.01, size=q_vals.shape)
        
        return int(np.argmax(q_vals))
#def select_action(state: np.ndarray, net: DQN, step: int, eps_start: float, eps_end: float, eps_decay: int) -> int:
#  this was suppose to help when i switched to test classic but stll had 0% 
#   eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * step / eps_decay)
#    if random.random() < eps:
#        return random.randrange(net.n_actions)
#    else:
#       with torch.no_grad():
#            q_values = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
#            return int(q_values.argmax())


# ───────────── Optimization with Prioritized Replay ─────────────
def optimise(memory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch: int, gamma: float,
             n_step: int = 1, n_step_buffer: Optional[deque] = None):
    """
    Optimize with support for prioritized replay and n-step returns.
    
    Args:
        memory: ReplayMemory or PrioritizedReplayMemory
        policy: Policy network
        target: Target network
        optimiser: Optimizer
        batch: Batch size
        gamma: Discount factor
        n_step: Number of steps for n-step returns (1 = standard)
        n_step_buffer: Buffer for n-step transitions (if n_step > 1)
    """
    if len(memory) < batch:
        return
    
    # Sample from memory (returns weights if prioritized)
    sample_result = memory.sample(batch)
    if isinstance(memory, PrioritizedReplayMemory):
        (s, a, r, s2, d), indices, weights = sample_result
        weights = torch.as_tensor(weights, device=DEVICE, dtype=torch.float32)
    else:
        (s, a, r, s2, d), indices, weights = sample_result
        weights = None
    
    s  = torch.as_tensor(s, device=DEVICE)
    s2 = torch.as_tensor(s2, device=DEVICE)
    a  = torch.as_tensor(a, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    r  = torch.as_tensor(r, device=DEVICE, dtype=torch.float32)
    d  = torch.as_tensor(d, device=DEVICE, dtype=torch.float32)

    q = policy(s).gather(1, a).squeeze(1)
    with torch.no_grad():
        # Double DQN: select action with policy, evaluate with target
        next_actions = policy(s2).argmax(dim=1, keepdim=True)   # shape (B,1)
        q2 = target(s2).gather(1, next_actions).squeeze(1)       # shape (B,)
        tgt = r + (1.0 - d) * gamma * q2

    # Compute TD errors for prioritized replay
    td_errors = (q - tgt).detach().cpu().numpy()
    
    # Weighted loss for prioritized replay
    if weights is not None:
        loss = (weights * nn.functional.smooth_l1_loss(q, tgt, reduction='none')).mean()
    else:
        loss = nn.functional.smooth_l1_loss(q, tgt)
    
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()
    policy.reset_noise()
    target.reset_noise()
    
    # Update priorities if using prioritized replay
    if isinstance(memory, PrioritizedReplayMemory) and indices is not None:
        memory.update_priorities(indices, td_errors)
