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
    """
    TODO: IMPLEMENT THIS CLASS
    
    Deep Q-Network using Convolutional Neural Networks for Pac-Man.
    
    CONCEPT:
    The DQN approximates the Q-function Q(state, action) using a neural network.
    For Pac-Man, the state is an RGB image of the game board, so we use CNNs
    to extract spatial features, then fully connected layers to output Q-values.
    
    ARCHITECTURE DESIGN:
    Your network should process game screenshots (RGB images) and output 
    Q-values for each possible action. Consider:
    - Convolutional layers to detect game features (walls, pellets, ghosts)
    - Pooling or strided convolutions to reduce spatial dimensions
    - Fully connected layers to combine features and output action values
    - Appropriate activation functions (ReLU is common)
    
    PARAMETERS for __init__:
    - obs_shape: Tuple[int, int, int] - Shape of input observations (H, W, C)
    - n_actions: int - Number of possible actions (4 for Pac-Man: up/down/left/right)
    
    METHODS TO IMPLEMENT:
    - __init__(self, obs_shape, n_actions): Initialize layers
    - forward(self, x): Forward pass through network
    
    INPUT FORMAT:
    - x arrives as (batch_size, H, W, C) uint8 tensor from environment
    - You need to convert to float, normalize to [0,1], and rearrange to (B, C, H, W)
    - PyTorch conv layers expect channels-first format: (batch, channels, height, width)
    
    OUTPUT FORMAT:
    - Return tensor of shape (batch_size, n_actions) with Q-values for each action
    
    
    TORCH FUNCTIONS YOU'LL NEED:
    - nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    - nn.ReLU()
    - nn.Flatten()
    - nn.Linear(in_features, out_features)
    - nn.Sequential() to chain layers
    - tensor.float() to convert to float
    - tensor.permute(0, 3, 1, 2) to rearrange dimensions
    - torch.zeros() for dummy input to calculate dimensions
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        # TODO: Implement network architecture
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        H, W, C = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            conv_flat_dim = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_flat_dim, 512), nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # dueling
        self.value_head = nn.Linear(512, 1) # V(s)
        self.adv_head = nn.Linear(512, n_actions) # A(s, a)
    
    def forward(self, x: torch.Tensor):
        """
        TODO: IMPLEMENT FORWARD PASS
        
        Process input through your network architecture.
        
        STEPS:
        1. Convert uint8 input to float and normalize by dividing by 255.0
        2. Rearrange from (B,H,W,C) to (B,C,H,W) using permute
        3. Pass through convolutional layers
        4. Pass through fully connected layers
        5. Return Q-values for each action
        """
        # TODO: Implement forward pass
        # x arrives as (B,H,W,C) uint8 from env; convert to float & channels‑first
        x = x.float().permute(0, 3, 1, 2) / 255.0

        feat = self.conv(x) # (B, 64, H, W)
        feat = feat.reshape(feat.size(0), -1) # (B, 64*H*W)
        h = self.fc(feat) # (B, 512)

        # Dueling
        value = self.value_head(h) # (B, 1)
        adv = self.adv_head(h) # B, n_actions)

        adv_mean = adv.mean(dim=1, keepdim=True) # (B, 1)
        q = value + (adv - adv_mean) # (B, n_actions)

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
def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int) -> int:
    """
    TODO: IMPLEMENT THIS FUNCTION
    
    Select an action using epsilon-greedy exploration strategy.
    
    CONCEPT:
    Epsilon-greedy balances exploration vs exploitation:
    - With probability epsilon: choose random action (exploration)
    - With probability (1-epsilon): choose action with highest Q-value (exploitation)
    - Epsilon should decay over time (start high for exploration, end low for exploitation)
    
    PARAMETERS:
    - state: np.ndarray - Current game state (image)
    - net: DQN - The neural network that estimates Q-values
    - step: int - Current training step (used for epsilon decay)
    - eps_start: float - Starting epsilon value (e.g., 1.0 = 100% random)
    - eps_end: float - Final epsilon value (e.g., 0.05 = 5% random)
    - eps_decay: int - How quickly epsilon decays (larger = slower decay)
    
    EPSILON DECAY:
    Use exponential decay: eps = eps_end + (eps_start - eps_end) * exp(-step / eps_decay)
    This gives you a smooth transition from high exploration to low exploration.
    
    IMPLEMENTATION STEPS:
    1. Calculate current epsilon using decay formula
    2. Generate random number and compare with epsilon
    3. If random < epsilon: return random action
    4. Else: use network to get Q-values and return best action
    
    TORCH FUNCTIONS YOU'LL NEED:
    - torch.as_tensor() to convert numpy to torch tensor
    - tensor.unsqueeze(0) to add batch dimension
    - torch.no_grad() context manager for inference
    - tensor.argmax() to find action with highest Q-value
    - int() to convert tensor to Python int
    
    RANDOM FUNCTIONS:
    - random.random() returns float in [0,1)
    - random.randrange(n) returns int in [0, n)
    - math.exp() for exponential function
    
    DEVICE HANDLING:
    - Move tensors to DEVICE before passing to network
    - Network is already on DEVICE, so inputs must match
    
    RETURN TYPE: int
    - Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
    """
    
    # TODO: Implement epsilon-greedy action selection
    eps = eps_end + (eps_start - eps_end) * math.exp(-step/eps_decay)

    if random.random() < eps:
        return random.randrange(net.n_actions)
    
    with torch.no_grad():
        s = torch.as_tensor(state, device=DEVICE).unsqueeze(0) # (1, H, W, 3)
        q_vals = net(s)
        action = int(q_vals.argmax(dim=1).item())

    return action

def optimise(memory: ReplayMemory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch_size: int, gamma: float):
    """
    TODO: IMPLEMENT THIS FUNCTION
    
    Perform one step of DQN optimization using experience replay and target network.
    This is the CORE of the DQN algorithm!
    
    CONCEPT:
    DQN uses two key innovations:
    1. Experience Replay: Store transitions and learn from random batches
    2. Target Network: Use separate network for computing targets (more stable)

    Recall discussion in class about the DQN update rule, and how we can write it as minimizing a loss function.
    
    PARAMETERS:
    - memory: ReplayMemory - Buffer containing stored experiences
    - policy: DQN - Main network being trained
    - target: DQN - Target network (updated less frequently)  
    - optimiser: torch optimizer - Updates policy network weights
    - batch_size: int - Number of experiences to sample
    - gamma: float - Discount factor for future rewards
    
    IMPLEMENTATION STEPS:
    1. Check if memory has enough experiences (return early if not)
    2. Sample batch of experiences: (state, action, reward, next_state, done)
    3. Convert numpy arrays to torch tensors on correct device
    4. Compute Q-values for current states using policy network
    5. Compute target Q-values for next states using target network
    6. Compute target values: reward + gamma * max_next_Q * (1 - done)
    7. Compute loss between current Q-values and targets
    8. Perform gradient descent step
    
    EXPERIENCE REPLAY SAMPLING:
    - memory.sample(batch_size) returns: states, actions, rewards, next_states, dones
    - Each is a numpy array with batch_size elements
    - Convert all to torch tensors on DEVICE
    
    COMPUTING CURRENT Q-VALUES:
    - Use policy(states) to get Q-values for all actions
    
    COMPUTING TARGET Q-VALUES:
    - Use torch.no_grad() to prevent gradients flowing to target network
    - Use target(next_states) to get Q-values for next states
    - Compute: rewards + (1 - dones) * gamma * max_next_q_values
    
    LOSS AND OPTIMIZATION:
    - Call optimiser.zero_grad() to clear old gradients
    - Call loss.backward() to compute gradients
    - Optionally clip gradients: nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    - Call optimiser.step() to update weights
    
    TENSOR OPERATIONS YOU'LL NEED:
    - torch.as_tensor(array, device=DEVICE, dtype=torch.float32)
    - tensor.unsqueeze(1) to add dimension
    - tensor.gather(1, indices) to select specific values
    - tensor.max(1)[0] to get maximum along dimension 1
    - tensor.squeeze() to remove extra dimensions
    
    RETURN TYPE: None (this function modifies policy network weights in-place)
    """
    
    # TODO: Implement DQN optimization step
    if len(memory) < batch_size:
        return
    
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states_t = torch.as_tensor(states, device=DEVICE, dtype=torch.float32)
    next_states_t = torch.as_tensor(next_states, device=DEVICE, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, device=DEVICE, dtype=torch.long)
    rewards_t = torch.as_tensor(rewards, device=DEVICE, dtype=torch.float32)
    dones_t = torch.as_tensor(dones, device=DEVICE, dtype=torch.float32)

    q_all = policy(states_t) # (B, n_actions)
    q_sa = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1) # (B,)

    with torch.no_grad():
        q_next_all = target(next_states_t) # (B, n_actions)
        q_next_max = q_next_all.max(dim=1)[0] # (B,)

        # target = rewards + gamma * max Q(s', a') * (1-done)
        targets = rewards_t + (1.0 - dones_t) * gamma * q_next_max # (B,)

    loss = nn.MSELoss()(q_sa, targets)

    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()
