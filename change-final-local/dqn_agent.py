# dqn_agent.py - Improved DQN with Frame Stacking, PER, Attention, N-Step Returns

from __future__ import annotations
import math
import random
from collections import deque
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# FRAME STACKING - Allows agent to infer motion/velocity
# ══════════════════════════════════════════════════════════════════════════════
class FrameStack:
    """Stack k consecutive frames to provide temporal information."""
    
    def __init__(self, k: int = 4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Initialize stack with copies of the first frame."""
        for _ in range(self.k):
            self.frames.append(frame)
        return np.concatenate(list(self.frames), axis=-1)
    
    def step(self, frame: np.ndarray) -> np.ndarray:
        """Add new frame and return stacked frames."""
        self.frames.append(frame)
        return np.concatenate(list(self.frames), axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED DUELING DQN WITH ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
class DQN(nn.Module):
    """
    Improved Dueling DQN with:
    - Deeper convolutional layers
    - Spatial attention mechanism
    - Residual-style connections
    - Larger fully connected heads
    """
    
    def __init__(self, obs_shape, n_actions: int, frame_stack: int = 4):
        super().__init__()
        
        # Input channels: 3 (RGB) * frame_stack
        in_channels = 3 * frame_stack
        
        # Feature extractor - deeper network
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Spatial attention module - helps focus on ghosts and pellets
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = 128
        
        # Dueling architecture - separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        self.adv_stream = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single observation (add batch dimension)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Normalize and convert to NCHW format
        x = x.to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Feature extraction with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Apply spatial attention
        att_weights = self.attention(x)
        x = x * att_weights
        
        # Global average pooling
        pooled = self.global_pool(x).view(x.size(0), -1)
        
        # Dueling: Q = V + (A - mean(A))
        V = self.value_stream(pooled)
        A = self.adv_stream(pooled)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        
        return Q


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITIZED EXPERIENCE REPLAY
# ══════════════════════════════════════════════════════════════════════════════
class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay buffer.
    Samples transitions with probability proportional to their TD-error.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent (0 = uniform, 1 = full prioritization)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
    
    def push(self, *transition):
        """Add a transition with maximum priority."""
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(tuple(transition))
        else:
            self.buffer[self.pos] = tuple(transition)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample a batch with prioritized probabilities."""
        N = len(self.buffer)
        if N == 0:
            return None, None, None
        
        # Calculate sampling probabilities
        probs = self.priorities[:N] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        weights = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
        
        # Gather samples
        samples = [self.buffer[i] for i in indices]
        batch = tuple(map(np.array, zip(*samples)))
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6  # Small constant to avoid zero priority
    
    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# N-STEP REPLAY MEMORY (Can be combined with PER)
# ══════════════════════════════════════════════════════════════════════════════
class NStepPrioritizedReplayMemory:
    """
    Combines N-step returns with Prioritized Experience Replay.
    N-step returns provide better credit assignment for delayed rewards.
    """
    
    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        
        # Temporary buffer for n-step calculation
        self.n_step_buffer = deque(maxlen=n_step)
    
    def _get_n_step_info(self):
        """Calculate n-step return and final state."""
        # Start from the last transition
        reward, next_state, done = self.n_step_buffer[-1][2:5]
        
        # Work backwards through the buffer
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2:5]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d
        
        return reward, next_state, done
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to n-step buffer, then to main buffer when ready."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Only add to main buffer when we have n transitions
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self._get_n_step_info()
            state, action = self.n_step_buffer[0][:2]
            
            max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
            
            if len(self.buffer) < self.capacity:
                self.buffer.append((state, action, reward, next_state, done))
            else:
                self.buffer[self.pos] = (state, action, reward, next_state, done)
            
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
        
        # If episode ends, flush remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                # Calculate partial n-step return
                reward_sum = 0
                gamma_power = 1.0
                final_next_state = self.n_step_buffer[-1][3]
                final_done = self.n_step_buffer[-1][4]
                
                for i, trans in enumerate(self.n_step_buffer):
                    reward_sum += gamma_power * trans[2]
                    gamma_power *= self.gamma
                    if trans[4]:  # done
                        final_next_state = trans[3]
                        final_done = True
                        break
                
                state, action = self.n_step_buffer[0][:2]
                
                max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
                
                if len(self.buffer) < self.capacity:
                    self.buffer.append((state, action, reward_sum, final_next_state, final_done))
                else:
                    self.buffer[self.pos] = (state, action, reward_sum, final_next_state, final_done)
                
                self.priorities[self.pos] = max_prio
                self.pos = (self.pos + 1) % self.capacity
                
                self.n_step_buffer.popleft()
    
    def sample(self, batch_size: int):
        """Sample with prioritization."""
        N = len(self.buffer)
        if N < batch_size:
            return None, None, None
        
        probs = self.priorities[:N] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
        
        samples = [self.buffer[i] for i in indices]
        batch = tuple(map(np.array, zip(*samples)))
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE REPLAY MEMORY (kept for compatibility)
# ══════════════════════════════════════════════════════════════════════════════
class ReplayMemory:
    """Simple uniform replay buffer."""
    
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    
    def push(self, *transition):
        self.buf.append(tuple(transition))
    
    def sample(self, batch_size: int):
        s = random.sample(self.buf, batch_size)
        return map(np.array, zip(*s))
    
    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# ACTION SELECTION WITH EPSILON-GREEDY
# ══════════════════════════════════════════════════════════════════════════════
def select_action(
    state: np.ndarray,
    net: DQN,
    step: int,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    n_actions: int
) -> int:
    """
    Select action using epsilon-greedy policy with exponential decay.
    """
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
    
    if random.random() < eps:
        return random.randrange(n_actions)
    
    with torch.no_grad():
        q = net(torch.as_tensor(state, device=DEVICE))
        return int(q.argmax())


def select_action_noisy(
    state: np.ndarray,
    net: DQN,
    step: int,
    n_actions: int,
    noise_scale: float = 0.1
) -> int:
    """
    Alternative: Select action with added noise to Q-values.
    Can help exploration in some cases.
    """
    with torch.no_grad():
        q = net(torch.as_tensor(state, device=DEVICE))
        noise = torch.randn_like(q) * noise_scale
        return int((q + noise).argmax())


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION STEP - Standard version
# ══════════════════════════════════════════════════════════════════════════════
def optimise(
    memory: ReplayMemory,
    policy: DQN,
    target: DQN,
    optimiser: optim.Optimizer,
    batch: int,
    gamma: float
):
    """Standard optimization step with uniform sampling."""
    if len(memory) < batch:
        return
    
    # Sample batch
    s, a, r, s2, d = memory.sample(batch)
    
    # Convert to tensors
    states = torch.as_tensor(s, device=DEVICE)
    next_states = torch.as_tensor(s2, device=DEVICE)
    actions = torch.as_tensor(a, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor(r, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    dones = torch.as_tensor(d, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    
    # Current Q-values
    q_values = policy(states).gather(1, actions)
    
    # Double DQN target
    with torch.no_grad():
        next_actions = policy(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions)
        target_q = rewards + (1.0 - dones) * gamma * next_q
    
    loss = F.smooth_l1_loss(q_values, target_q)
    
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimiser.step()


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION STEP - Prioritized version
# ══════════════════════════════════════════════════════════════════════════════
def optimise_prioritized(
    memory: PrioritizedReplayMemory,
    policy: DQN,
    target: DQN,
    optimiser: optim.Optimizer,
    batch: int,
    gamma: float
) -> Optional[float]:
    """Optimization step with prioritized experience replay."""
    if len(memory) < batch:
        return None
    
    # Sample batch with priorities
    sample_result = memory.sample(batch)
    if sample_result[0] is None:
        return None
    
    (s, a, r, s2, d), indices, weights = sample_result
    
    # Convert to tensors
    states = torch.as_tensor(s, device=DEVICE)
    next_states = torch.as_tensor(s2, device=DEVICE)
    actions = torch.as_tensor(a, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor(r, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    dones = torch.as_tensor(d, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    
    # Current Q-values
    q_values = policy(states).gather(1, actions)
    
    # Double DQN target
    with torch.no_grad():
        next_actions = policy(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions)
        target_q = rewards + (1.0 - dones) * gamma * next_q
    
    # Calculate TD-errors for priority update
    td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
    
    # Weighted loss (importance sampling correction)
    elementwise_loss = F.smooth_l1_loss(q_values, target_q, reduction='none')
    loss = (elementwise_loss * weights.unsqueeze(1)).mean()
    
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimiser.step()
    
    # Update priorities
    memory.update_priorities(indices, td_errors)
    
    return loss.item()


def optimise_nstep_prioritized(
    memory: NStepPrioritizedReplayMemory,
    policy: DQN,
    target: DQN,
    optimiser: optim.Optimizer,
    batch: int,
    gamma: float,
    n_step: int = 3
) -> Optional[float]:
    """Optimization step with N-step prioritized experience replay."""
    if len(memory) < batch:
        return None
    
    sample_result = memory.sample(batch)
    if sample_result[0] is None:
        return None
    
    (s, a, r, s2, d), indices, weights = sample_result
    
    # Convert to tensors
    states = torch.as_tensor(s, device=DEVICE)
    next_states = torch.as_tensor(s2, device=DEVICE)
    actions = torch.as_tensor(a, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor(r, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    dones = torch.as_tensor(d, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    
    # Current Q-values
    q_values = policy(states).gather(1, actions)
    
    # N-step Double DQN target
    # Note: rewards already contain n-step discounted sum from the memory
    with torch.no_grad():
        next_actions = policy(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions)
        # gamma^n for the bootstrap value (rewards already has gamma^0 to gamma^(n-1))
        gamma_n = gamma ** n_step
        target_q = rewards + (1.0 - dones) * gamma_n * next_q
    
    # Calculate TD-errors for priority update
    td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
    
    # Weighted loss
    elementwise_loss = F.smooth_l1_loss(q_values, target_q, reduction='none')
    loss = (elementwise_loss * weights.unsqueeze(1)).mean()
    
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimiser.step()
    
    # Update priorities
    memory.update_priorities(indices, td_errors)
    
    return loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def augment_transition(
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: float
) -> List[Tuple]:
    """
    Augment a transition with horizontal flip.
    Returns list of (state, action, reward, next_state, done) tuples.
    """
    transitions = [(state, action, reward, next_state, done)]
    
    # Horizontal flip with 50% probability
    if random.random() < 0.5:
        s_flip = np.flip(state, axis=1).copy()
        ns_flip = np.flip(next_state, axis=1).copy()
        # Map actions: left(2) <-> right(3), up(0) and down(1) unchanged
        action_map = {0: 0, 1: 1, 2: 3, 3: 2}
        transitions.append((s_flip, action_map[action], reward, ns_flip, done))
    
    return transitions

