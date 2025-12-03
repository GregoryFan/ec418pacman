# Pac-Man DQN - Advanced Implementation

This implementation includes several advanced techniques to maximize win rate across all layouts.

## Features Implemented

### Network Architecture
- **Dueling DQN**: Separate value and advantage streams for better learning
- **Deeper CNN**: 4 convolutional layers with batch normalization
- **Spatial Attention**: Focuses on relevant parts of the game state (ghosts, pellets)
- **He Initialization**: Proper weight initialization for ReLU networks

### Training Techniques
- **Frame Stacking (4 frames)**: Allows agent to infer motion/velocity of ghosts
- **Prioritized Experience Replay**: Samples important transitions more frequently
- **N-Step Returns (3 steps)**: Better credit assignment for delayed rewards
- **Double DQN**: Reduces overestimation of Q-values
- **Data Augmentation**: Horizontal flips for additional training data

### Training Strategies
- **Reward Shaping**: Intermediate rewards for approaching pellets and avoiding ghosts
- **Curriculum Learning**: Progressive training from easier to harder layouts
- **Multi-task Training**: Train on all layouts simultaneously for generalization
- **Learning Rate Scheduling**: Cosine annealing for stable convergence

## Files

| File | Description |
|------|-------------|
| `dqn_agent.py` | Neural network, replay buffers, optimization functions |
| `pacman_env.py` | Gymnasium environment with reward shaping |
| `train.py` | Training script with multiple modes |
| `play_cv.py` | Visualization and evaluation script |
| `play_pacman.py` | Alternative visualization script |

## Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy gymnasium matplotlib opencv-python
```

### 2. Training

**Curriculum Learning (Recommended):**
```bash
python train.py --mode curriculum --episodes 1000
```
This trains progressively: spiral → spiral_harder → empty → classic → multi-task

**Multi-task Training:**
```bash
python train.py --mode multitask --episodes 1500
```

**Single Layout:**
```bash
python train.py --mode single --layout classic --episodes 2000
```

**Quick Test Run:**
```bash
python train.py --mode curriculum --fast
```

### 3. Evaluation

**Full Competition Evaluation:**
```bash
python play_cv.py --eval-all --model pacman_dqn_multitask.pt --episodes 50
```

**Single Layout:**
```bash
python play_cv.py --layout classic --episodes 20
```

**Headless (no display):**
```bash
python play_cv.py --layout classic --headless --episodes 50
```

### 4. Visualization

**Watch Agent Play:**
```bash
python play_cv.py --layout classic --scale 4 --speed 80
```

**Controls during visualization:**
- `q`: Quit
- `Space`: Pause/Resume
- `+`/`-`: Speed up/Slow down

## Training Tips

1. **Start with Curriculum Learning**: It provides the best results by building skills progressively.

2. **Training Time**: 
   - Fast mode: ~30 minutes
   - Full curriculum: 2-4 hours
   - With GPU: 3-4x faster

3. **Monitor Progress**: Check win rate every 50-100 episodes. Good models should achieve:
   - spiral: 90%+
   - spiral_harder: 70%+
   - empty: 60%+
   - classic: 40%+

4. **If Training Stalls**: Try adjusting:
   - Increase `EPS_DECAY` for slower exploration decay
   - Lower learning rate
   - Increase batch size

## Expected Results

After full curriculum training (~4000 total episodes):

| Layout | Expected Win Rate |
|--------|-------------------|
| spiral | 85-95% |
| spiral_harder | 65-80% |
| empty | 55-70% |
| classic | 35-50% |
| **Average** | **60-75%** |

## Hyperparameter Reference

```python
# Key hyperparameters in train.py
FRAME_STACK = 4          # Number of stacked frames
N_STEP = 3               # N-step returns
BATCH_SIZE = 64          # Batch size for training
MEMORY_CAP = 50000       # Replay buffer size
GAMMA = 0.99             # Discount factor
LR = 5e-4                # Learning rate
TARGET_FREQ = 500        # Target network update frequency
```

## Troubleshooting

**CUDA out of memory:**
- Reduce `BATCH_SIZE` to 32
- Reduce `MEMORY_CAP` to 20000

**Training too slow:**
- Use `--fast` flag for quick testing
- Ensure GPU is being used (check "Device: cuda" in output)

**Low win rate:**
- Train for more episodes
- Try curriculum learning instead of single-layout training
- Check if reward shaping is enabled in environment

## Competition Scoring

Final score is calculated as:
```
Score = (Win_classic + Win_spiral + Win_spiral_harder + Win_empty) / 4
```

Use `python play_cv.py --eval-all --episodes 50` to compute this score.