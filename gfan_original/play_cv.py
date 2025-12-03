#!/usr/bin/env python
"""
play_cv.py – visualise a trained Pac‑Man agent with OpenCV animation.
Usage example:
    python play_cv.py --layout classic --scale 3 --speed 70
    python play_cv.py --layout classic --headless  # No display
"""

from __future__ import annotations
import argparse, sys, os, random
from pathlib import Path
import torch
import numpy as np
from pacman_env import PacmanEnv
#from dqn_agent import DuelingDQN as DQN, DEVICE
from dqn_noisy import DQN, select_action, DEVICE

# Try to import cv2, but handle gracefully if display is not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - running in headless mode")

# ───────────────────────── helper ─────────────────────────
#def load_net(weight_file: Path, n_actions: int, obs_shape) -> DQN:
#   net = DQN(obs_shape, n_actions).to(DEVICE)
#    net.load_state_dict(torch.load(weight_file, map_location=DEVICE))
#    net.eval()
#  return net

def load_net(weight_file: Path, n_actions: int, obs_shape) -> DQN:
    """
    Robust loader that inspects the checkpoint to decide whether to build the
    deeper DQN variant (the one used to create the saved weights).
    """
    # 1) Read the checkpoint dict (state_dict) first
    ckpt = torch.load(weight_file, map_location=DEVICE)
    # ckpt might be either a plain state_dict or a dict with extra metadata
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 2) Inspect keys to guess whether the checkpoint used the deeper model
    # deeper model introduces extra conv layer -> keys like 'conv.6.weight' appear,
    # and NoisyLinear parameter names (weight_mu, weight_sigma) are present.
    deeper_flag = False
    if any(k.startswith("conv.6") or ".conv.6" in k for k in state_dict.keys()):
        deeper_flag = True
    else:
        # also check for value/adv extra-layer shapes in parameter names
        if any("value_stream.4.weight_mu" in k or "adv_stream.4.weight_mu" in k for k in state_dict.keys()):
            deeper_flag = True

    if deeper_flag:
        print("Checkpoint appears to be from the deeper DQN architecture -> building DQN(..., deeper=True)")
    else:
        print("Checkpoint appears to be from the shallow DQN architecture -> building DQN(..., deeper=False)")

    # 3) Instantiate model accordingly
    net = DQN(obs_shape, n_actions, deeper=deeper_flag).to(DEVICE)

    # 4) Try strict load first; if it fails, try strict=False and show diagnostics
    try:
        net.load_state_dict(state_dict)
        print("Loaded state_dict with strict=True")
    except RuntimeError as e:
        print("Strict load failed (shapes/keys mismatch). Attempting partial load with strict=False...")
        # Show a short summary of missing/unexpected keys for debugging
        try:
            missing, unexpected = net.load_state_dict(state_dict, strict=False)
        except Exception:
            # Different torch versions return different exceptions / formats
            # Fall back to printing the runtime error and then try non-strict load anyway
            print("Warning: detailed strict=False diagnostic unavailable - attempting non-strict load.")
            net.load_state_dict(state_dict, strict=False)
        print("Partial load complete (strict=False).")
        # Print a little hint for the user
        print("If performance is poor, ensure the DQN code used for training and playing match exactly.")
    net.eval()
    return net


def is_display_available():
    """Check if display is available for OpenCV."""
    if not CV2_AVAILABLE:
        return False
    
    # Only check DISPLAY for Linux (not macOS)
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    
    # Try to create a test window
    try:
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False

# ───────────────────────── visual play loop ──────────────────────
def play_visual(layout: str, net: DQN, episodes: int, delay_ms: int, scale: int):
    """Play with OpenCV visualization."""
    env = PacmanEnv(layout)
   

    wins = 0
    cv2.namedWindow("Pac-Man", cv2.WINDOW_AUTOSIZE)  # Also changed the dash
    cv2.waitKey(1)  # Let macOS initialize the window

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, step, win = False, 0, False

        while not done and step < 1000:
            # Use select_action with tie-breaking (training=False for greedy, but still breaks ties)
            action = select_action(state, net, training=False, epsilon=0.0)
            state, _, done, _, _ = env.step(action)
            step += 1
            if done and not env.pellets:
                win = True

            frame = cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR)
            if scale > 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame, f"{layout}  Ep {ep}/{episodes}  step {step}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.imshow("Pac‑Man", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                print("Quit requested.")
                env.close(); cv2.destroyAllWindows(); sys.exit(0)

        wins += win
        print(f"Episode {ep}: {'WIN' if win else 'LOSE'} in {step} steps")

    print(f"\n{wins}/{episodes} wins ({wins/episodes*100:.1f}%)")
    env.close(); cv2.destroyAllWindows()

# ───────────────────────── headless play loop ──────────────────────
def play_headless(layout: str, net: DQN, episodes: int):
    """Play without display - text output only."""
    env = PacmanEnv(layout)
    

    wins = 0
    
    print(f"Running DQN agent on {layout} layout (headless mode)")
    print(f"Device: {DEVICE}")
    print("-" * 50)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, step, win = False, 0, False
        
        print(f"\nEpisode {ep}/{episodes}: Starting game")
        initial_pellets = len(env.pellets)
        print(f"  Initial pellets: {initial_pellets}")
        print(f"  Starting position: Pac-Man at {env.pac_pos}, Ghost(s) at {env.ghost_pos}")

        last_action = None
        action_repeat_count = 0
        
        while not done and step < 1000:
            # Use select_action with tie-breaking
            with torch.no_grad():
                q_values = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
                q_vals = q_values.squeeze().cpu().numpy()
            
            # Break ties if Q-values are too similar
            if q_vals.max() - q_vals.min() < 0.1:
                q_vals = q_vals + np.random.normal(0, 0.01, size=q_vals.shape)
            action = int(np.argmax(q_vals))
            
            # If stuck repeating same action, force random exploration
            if action == last_action:
                action_repeat_count += 1
                if action_repeat_count > 10:
                    action = random.randrange(4)
                    action_repeat_count = 0
            else:
                action_repeat_count = 0
            last_action = action
            
            # Print action details for first few steps or periodically
            if step < 5 or step % 100 == 0:
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                print(f"  Step {step}: Action={actions[action]}, "
                      f"Q-values={[f'{q:.2f}' for q in q_values.squeeze().cpu().numpy()]}, "
                      f"Pellets left: {len(env.pellets)}")
            
            state, reward, done, _, _ = env.step(action)
            step += 1
            
            if done and not env.pellets:
                win = True

        wins += win
        pellets_collected = initial_pellets - len(env.pellets)
        result = "WIN" if win else "LOSE"
        print(f"Episode {ep}: {result} in {step} steps")
        print(f"  Pellets collected: {pellets_collected}/{initial_pellets}")
        print(f"  Final position: Pac-Man at {env.pac_pos}, Ghost(s) at {env.ghost_pos}")

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"Layout: {layout}")
    print(f"Model device: {DEVICE}")
    print(f"{'='*50}")
    
    env.close()

# ───────────────────────── main play function ──────────────────────
def play(layout: str, net: DQN, episodes: int, delay_ms: int = 100, scale: int = 3, headless: bool = False):
    """Play with automatic display detection or explicit headless mode."""
    if headless or not is_display_available():
        if not headless:
            print("No display available - running in headless mode")
        play_headless(layout, net, episodes)
    else:
        play_visual(layout, net, episodes, delay_ms, scale)

# ───────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout",
                        choices=["classic", "empty", "spiral", "spiral_harder"],
                        required=True)
    parser.add_argument("--model", type=Path,
                        help="explicit path to .pt weight file "
                             "(default pacman_dqn_<layout>.pt)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--speed", type=int, default=100,
                        help="delay between frames in ms (visual mode only)")
    parser.add_argument("--scale", type=int, default=3,
                        help="integer up‑scale factor for window size (visual mode only)")
    parser.add_argument("--headless", action="store_true",
                        help="run without display (text output only)")
    args = parser.parse_args()

    weight_path = args.model or Path(f"pacman_dqn_{args.layout}.pt")
    if not weight_path.exists():
        sys.exit(f"weight file {weight_path} not found")

    # Build a temp env to discover observation shape & action count
    tmp_env = PacmanEnv(args.layout)
    obs_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()



    net = load_net(weight_path, n_actions, obs_shape)
    play(args.layout, net, args.episodes, args.speed, args.scale, args.headless)