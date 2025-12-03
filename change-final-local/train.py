#!/usr/bin/env python
"""
train.py - Advanced DQN Training with:
- Frame Stacking (temporal information)
- Prioritized Experience Replay
- N-Step Returns
- Multi-task Training
- Curriculum Learning
- Data Augmentation
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from pacman_env import PacmanEnv
from dqn_agent import (
    DQN,
    FrameStack,
    NStepPrioritizedReplayMemory,
    PrioritizedReplayMemory,
    ReplayMemory,
    select_action,
    optimise,
    optimise_prioritized,
    optimise_nstep_prioritized,
    augment_transition,
    DEVICE
)


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    """Training configuration."""
    # Training episodes
    NUM_EPISODES = 2000
    NUM_EPISODES_FAST = 500
    
    # Network updates
    TARGET_FREQ = 500          # Steps between target network updates
    BATCH_SIZE = 64            # Smaller batch with PER
    MEMORY_CAP = 50000         # Larger replay buffer
    
    # Learning
    GAMMA = 0.99               # Discount factor
    LR = 5e-4                  # Learning rate
    LR_MIN = 1e-5              # Minimum learning rate
    
    # Exploration schedule
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 15000          # Slower decay
    
    # Fine-tuning (transfer learning)
    FINE_TUNE_EPS_START = 0.3
    FINE_TUNE_EPS_END = 0.02
    FINE_TUNE_EPS_DECAY = 10000
    
    # Frame stacking
    FRAME_STACK = 4
    
    # N-step returns
    N_STEP = 3
    
    # PER parameters
    PER_ALPHA = 0.6            # Prioritization exponent
    PER_BETA_START = 0.4       # Initial importance sampling
    PER_BETA_FRAMES = 100000   # Frames to anneal beta to 1.0
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Multi-task
    MULTITASK_SAMPLE_OTHER = True  # Sample from other layout buffers


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE LAYOUT TRAINER (with all improvements)
# ═══════════════════════════════════════════════════════════════════════════════
def train_layout(
    layout: str,
    episodes: int,
    init_weights: Path | None = None,
    eps_start: float = Config.EPS_START,
    eps_end: float = Config.EPS_END,
    eps_decay: int = Config.EPS_DECAY,
    use_nstep: bool = True,
    use_per: bool = True,
    use_augmentation: bool = True,
    save_every: int = 500,
) -> Path:
    """
    Train DQN on a single layout with advanced techniques.
    """
    print(f"\n{'='*60}")
    print(f"Training on layout: {layout}")
    print(f"Episodes: {episodes}")
    print(f"Device: {DEVICE}")
    print(f"Using N-Step: {use_nstep}, PER: {use_per}, Augmentation: {use_augmentation}")
    print(f"{'='*60}\n")

    # Create environment
    env = PacmanEnv(layout, reward_shaping=True)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Create networks
    policy = DQN(obs_shape, n_actions, frame_stack=Config.FRAME_STACK).to(DEVICE)
    target = DQN(obs_shape, n_actions, frame_stack=Config.FRAME_STACK).to(DEVICE)

    # Load pretrained weights if provided
    if init_weights is not None:
        print(f"Loading weights from {init_weights}")
        state_dict = torch.load(init_weights, map_location=DEVICE)
        policy.load_state_dict(state_dict)
    
    target.load_state_dict(policy.state_dict())
    target.eval()

    # Optimizer with scheduler
    optimiser = optim.Adam(policy.parameters(), lr=Config.LR)
    scheduler = CosineAnnealingLR(optimiser, T_max=episodes, eta_min=Config.LR_MIN)

    # Choose replay buffer type
    if use_nstep and use_per:
        memory = NStepPrioritizedReplayMemory(
            Config.MEMORY_CAP,
            n_step=Config.N_STEP,
            gamma=Config.GAMMA,
            alpha=Config.PER_ALPHA,
            beta_start=Config.PER_BETA_START,
            beta_frames=Config.PER_BETA_FRAMES
        )
        optimise_fn = lambda: optimise_nstep_prioritized(
            memory, policy, target, optimiser, Config.BATCH_SIZE, Config.GAMMA, Config.N_STEP
        )
    elif use_per:
        memory = PrioritizedReplayMemory(
            Config.MEMORY_CAP,
            alpha=Config.PER_ALPHA,
            beta_start=Config.PER_BETA_START,
            beta_frames=Config.PER_BETA_FRAMES
        )
        optimise_fn = lambda: optimise_prioritized(
            memory, policy, target, optimiser, Config.BATCH_SIZE, Config.GAMMA
        )
    else:
        memory = ReplayMemory(Config.MEMORY_CAP)
        optimise_fn = lambda: optimise(
            memory, policy, target, optimiser, Config.BATCH_SIZE, Config.GAMMA
        )

    # Frame stacker
    frame_stack = FrameStack(k=Config.FRAME_STACK)

    # Training loop
    step = 0
    rewards_history = []
    wins_history = []
    best_avg_reward = float('-inf')

    for ep in range(1, episodes + 1):
        # Reset environment and frame stack
        raw_state, _ = env.reset()
        state = frame_stack.reset(raw_state)
        
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # Select action
            action = select_action(
                state, policy, step,
                eps_start, eps_end, eps_decay,
                n_actions
            )
            step += 1

            # Execute action
            raw_next_state, reward, done, _, _ = env.step(action)
            next_state = frame_stack.step(raw_next_state)

            # Store transition (with optional augmentation)
            if use_augmentation and random.random() < 0.3:
                transitions = augment_transition(state, action, reward, next_state, float(done))
                for trans in transitions:
                    memory.push(*trans)
            else:
                memory.push(state, action, reward, next_state, float(done))

            state = next_state
            ep_reward += reward
            ep_steps += 1

            # Optimize
            if len(memory) >= Config.BATCH_SIZE:
                optimise_fn()

            # Update target network
            if step % Config.TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        # Track metrics
        rewards_history.append(ep_reward)
        won = len(env.pellets) == 0
        wins_history.append(won)

        # Update learning rate
        scheduler.step()

        # Logging
        if ep % 50 == 0 or ep == episodes:
            recent_rewards = rewards_history[-50:]
            recent_wins = wins_history[-50:]
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins) * 100
            current_lr = scheduler.get_last_lr()[0]
            eps = eps_end + (eps_start - eps_end) * np.exp(-step / eps_decay)
            
            print(f"[{layout}] Ep {ep:4d} | "
                  f"Reward: {ep_reward:7.1f} | "
                  f"Avg(50): {avg_reward:7.1f} | "
                  f"WinRate: {win_rate:5.1f}% | "
                  f"Steps: {ep_steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"LR: {current_lr:.2e}")

            # Save best model
            if avg_reward > best_avg_reward and ep > 100:
                best_avg_reward = avg_reward
                best_path = Path(f"pacman_dqn_{layout}_best.pt")
                torch.save(policy.state_dict(), best_path)
                print(f"  → New best model saved: {best_path}")

        # Periodic checkpoint
        if save_every > 0 and ep % save_every == 0:
            ckpt_path = Path(f"pacman_dqn_{layout}_ep{ep}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    env.close()

    # Save final model
    weight_path = Path(f"pacman_dqn_{layout}.pt")
    torch.save(policy.state_dict(), weight_path)
    
    # Print summary
    final_win_rate = np.mean(wins_history[-100:]) * 100
    print(f"\n[{layout}] Training complete!")
    print(f"  Final win rate (last 100): {final_win_rate:.1f}%")
    print(f"  Saved to: {weight_path.resolve()}")
    
    return weight_path


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TASK TRAINER (train on all layouts simultaneously)
# ═══════════════════════════════════════════════════════════════════════════════
def train_multitask(
    layouts: list[str],
    episodes_per_layout: int,
    init_weights: Path | None = None,
    layout_weights: dict[str, float] | None = None,
) -> Path:
    """
    Train a single network on multiple layouts simultaneously.
    Uses separate replay buffers per layout.
    """
    print(f"\n{'='*60}")
    print(f"Multi-task training on layouts: {layouts}")
    print(f"Episodes per layout: {episodes_per_layout}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Default layout weights (how often to sample each)
    if layout_weights is None:
        layout_weights = {
            "classic": 0.4,       # Hardest, needs more training
            "spiral_harder": 0.25,
            "spiral": 0.15,
            "empty": 0.2,
        }

    # Create environments
    envs = {layout: PacmanEnv(layout, reward_shaping=True) for layout in layouts}
    n_actions = 4
    obs_shape = envs[layouts[0]].observation_space.shape

    # Create single shared network
    policy = DQN(obs_shape, n_actions, frame_stack=Config.FRAME_STACK).to(DEVICE)
    target = DQN(obs_shape, n_actions, frame_stack=Config.FRAME_STACK).to(DEVICE)

    if init_weights is not None:
        print(f"Loading weights from {init_weights}")
        state_dict = torch.load(init_weights, map_location=DEVICE)
        policy.load_state_dict(state_dict)

    target.load_state_dict(policy.state_dict())
    target.eval()

    # Optimizer
    optimiser = optim.Adam(policy.parameters(), lr=Config.LR)
    total_episodes = episodes_per_layout * len(layouts)
    scheduler = CosineAnnealingLR(optimiser, T_max=total_episodes, eta_min=Config.LR_MIN)

    # Separate replay buffers per layout
    memories = {
        layout: NStepPrioritizedReplayMemory(
            Config.MEMORY_CAP // len(layouts),
            n_step=Config.N_STEP,
            gamma=Config.GAMMA,
        )
        for layout in layouts
    }

    # Frame stackers per layout
    frame_stacks = {layout: FrameStack(k=Config.FRAME_STACK) for layout in layouts}

    # Metrics per layout
    metrics = {
        layout: {"rewards": [], "wins": [], "episodes": 0}
        for layout in layouts
    }

    step = 0
    
    for ep in range(1, total_episodes + 1):
        # Sample layout based on weights
        layout = random.choices(
            layouts,
            weights=[layout_weights.get(l, 1.0/len(layouts)) for l in layouts]
        )[0]
        
        env = envs[layout]
        memory = memories[layout]
        frame_stack = frame_stacks[layout]

        # Reset
        raw_state, _ = env.reset()
        state = frame_stack.reset(raw_state)
        
        done = False
        ep_reward = 0.0

        while not done:
            action = select_action(
                state, policy, step,
                Config.EPS_START, Config.EPS_END, Config.EPS_DECAY,
                n_actions
            )
            step += 1

            raw_next_state, reward, done, _, _ = env.step(action)
            next_state = frame_stack.step(raw_next_state)

            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            # Optimize from current layout's buffer
            if len(memory) >= Config.BATCH_SIZE:
                optimise_nstep_prioritized(
                    memory, policy, target, optimiser,
                    Config.BATCH_SIZE, Config.GAMMA, Config.N_STEP
                )

            # Occasionally sample from other layouts' buffers
            if Config.MULTITASK_SAMPLE_OTHER and step % 4 == 0:
                other_layouts = [l for l in layouts if l != layout]
                for other in other_layouts:
                    other_mem = memories[other]
                    if len(other_mem) >= Config.BATCH_SIZE // 2:
                        optimise_nstep_prioritized(
                            other_mem, policy, target, optimiser,
                            Config.BATCH_SIZE // 2, Config.GAMMA, Config.N_STEP
                        )

            if step % Config.TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        # Track metrics
        metrics[layout]["rewards"].append(ep_reward)
        metrics[layout]["wins"].append(len(env.pellets) == 0)
        metrics[layout]["episodes"] += 1

        scheduler.step()

        # Logging
        if ep % 100 == 0 or ep == total_episodes:
            print(f"\n[Multi-task] Episode {ep}/{total_episodes}")
            for l in layouts:
                if metrics[l]["episodes"] > 0:
                    recent = min(50, len(metrics[l]["rewards"]))
                    avg_r = np.mean(metrics[l]["rewards"][-recent:])
                    win_r = np.mean(metrics[l]["wins"][-recent:]) * 100
                    print(f"  {l:15s}: Eps={metrics[l]['episodes']:4d}, "
                          f"AvgReward={avg_r:7.1f}, WinRate={win_r:5.1f}%")

    # Close environments
    for env in envs.values():
        env.close()

    # Save model
    weight_path = Path("pacman_dqn_multitask.pt")
    torch.save(policy.state_dict(), weight_path)
    print(f"\n[Multi-task] Training complete! Saved to: {weight_path.resolve()}")

    # Also save per-layout copies
    for layout in layouts:
        layout_path = Path(f"pacman_dqn_{layout}.pt")
        torch.save(policy.state_dict(), layout_path)
        print(f"  → Also saved as: {layout_path}")

    return weight_path


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM LEARNING (progressive difficulty)
# ═══════════════════════════════════════════════════════════════════════════════
def train_curriculum(episodes_per_phase: int = 1000) -> Path:
    """
    Curriculum learning: start with easier layouts, transfer to harder ones.
    
    Phase 1: spiral (simple structure, single ghost)
    Phase 2: spiral_harder (same structure, aggressive ghost)
    Phase 3: empty (open space, need different strategy)
    Phase 4: classic (full maze, multiple ghosts)
    """
    print("\n" + "="*60)
    print("CURRICULUM LEARNING")
    print("="*60)

    # Phase 1: Learn basic movement on spiral
    print("\n>>> Phase 1: spiral (basic navigation)")
    w1 = train_layout(
        "spiral",
        episodes=episodes_per_phase,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=10000,
        save_every=0,
    )

    # Phase 2: Transfer to spiral_harder
    print("\n>>> Phase 2: spiral_harder (aggressive ghost)")
    w2 = train_layout(
        "spiral_harder",
        episodes=episodes_per_phase,
        init_weights=w1,
        eps_start=0.5,
        eps_end=0.05,
        eps_decay=8000,
        save_every=0,
    )

    # Phase 3: Transfer to empty (different strategy needed)
    print("\n>>> Phase 3: empty (open space)")
    w3 = train_layout(
        "empty",
        episodes=episodes_per_phase,
        init_weights=w2,
        eps_start=0.4,
        eps_end=0.05,
        eps_decay=8000,
        save_every=0,
    )

    # Phase 4: Final training on classic (hardest)
    print("\n>>> Phase 4: classic (full maze) - Extended training")
    w4 = train_layout(
        "classic",
        episodes=episodes_per_phase * 2,  # Extra episodes for hardest layout
        init_weights=w3,
        eps_start=0.3,
        eps_end=0.02,
        eps_decay=15000,
        save_every=500,
    )

    # Phase 5: Multi-task fine-tuning on all layouts
    print("\n>>> Phase 5: Multi-task fine-tuning")
    final_weights = train_multitask(
        layouts=["classic", "spiral", "spiral_harder", "empty"],
        episodes_per_layout=500,
        init_weights=w4,
        layout_weights={"classic": 0.35, "spiral_harder": 0.25, "empty": 0.25, "spiral": 0.15}
    )

    print("\n" + "="*60)
    print("CURRICULUM TRAINING COMPLETE!")
    print("="*60)
    
    return final_weights


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(
    weight_path: Path,
    layouts: list[str],
    episodes_per_layout: int = 50
) -> dict[str, float]:
    """
    Evaluate a trained model on multiple layouts.
    Returns win rates per layout.
    """
    print(f"\nEvaluating {weight_path}")
    print("-" * 40)

    results = {}

    for layout in layouts:
        env = PacmanEnv(layout, reward_shaping=False)  # No shaping during eval
        n_actions = env.action_space.n
        obs_shape = env.observation_space.shape

        policy = DQN(obs_shape, n_actions, frame_stack=Config.FRAME_STACK).to(DEVICE)
        policy.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        policy.eval()

        frame_stack = FrameStack(k=Config.FRAME_STACK)
        wins = 0

        for ep in range(episodes_per_layout):
            raw_state, _ = env.reset()
            state = frame_stack.reset(raw_state)
            done = False
            steps = 0

            while not done and steps < 1000:
                with torch.no_grad():
                    q = policy(torch.as_tensor(state, device=DEVICE))
                    action = int(q.argmax())

                raw_next_state, _, done, _, _ = env.step(action)
                state = frame_stack.step(raw_next_state)
                steps += 1

            if done and len(env.pellets) == 0:
                wins += 1

        env.close()
        win_rate = wins / episodes_per_layout * 100
        results[layout] = win_rate
        print(f"  {layout:15s}: {wins}/{episodes_per_layout} ({win_rate:.1f}%)")

    avg_win_rate = np.mean(list(results.values()))
    print(f"\n  {'Average':15s}: {avg_win_rate:.1f}%")
    results["average"] = avg_win_rate

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pac-Man")
    parser.add_argument(
        "--mode",
        choices=["single", "multitask", "curriculum", "eval"],
        default="curriculum",
        help="Training mode"
    )
    parser.add_argument(
        "--layout",
        choices=["classic", "empty", "spiral", "spiral_harder"],
        default="classic",
        help="Layout for single mode"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick training run"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Initial weights for fine-tuning or evaluation"
    )
    parser.add_argument(
        "--no-per",
        action="store_true",
        help="Disable prioritized experience replay"
    )
    parser.add_argument(
        "--no-nstep",
        action="store_true",
        help="Disable N-step returns"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )

    args = parser.parse_args()

    # Determine episode count
    if args.episodes:
        episodes = args.episodes
    elif args.fast:
        episodes = Config.NUM_EPISODES_FAST
    else:
        episodes = Config.NUM_EPISODES

    print(f"PyTorch device: {DEVICE}")
    print(f"Frame stacking: {Config.FRAME_STACK} frames")
    print(f"N-step returns: {Config.N_STEP} steps")

    if args.mode == "single":
        train_layout(
            args.layout,
            episodes,
            init_weights=args.weights,
            use_per=not args.no_per,
            use_nstep=not args.no_nstep,
            use_augmentation=not args.no_augment,
        )

    elif args.mode == "multitask":
        train_multitask(
            layouts=["classic", "spiral", "spiral_harder", "empty"],
            episodes_per_layout=episodes,
            init_weights=args.weights,
        )

    elif args.mode == "curriculum":
        train_curriculum(episodes_per_phase=episodes)

    elif args.mode == "eval":
        if args.weights is None:
            args.weights = Path("pacman_dqn_multitask.pt")
        
        if not args.weights.exists():
            print(f"Error: weights file {args.weights} not found")
            exit(1)
        
        evaluate(
            args.weights,
            layouts=["classic", "spiral", "spiral_harder", "empty"],
            episodes_per_layout=50
        )