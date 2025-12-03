# pacman_env.py - Fixed with Portal Mechanics and Improved Reward Shaping
from __future__ import annotations
import math
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# rendering constants 
PIXELS_PER_CELL = 12  # tile size

# 7Ã7 boards 
tiny_walls = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 0, 0, 0, 0, 0, 1],
     [1, 0, 1, 1, 1, 0, 1],
     [1, 0, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 0, 0, 1],
     [1, 0, 0, 0, 1, 0, 1],
     [1, 1, 1, 1, 1, 1, 1]], dtype=int)

LAYOUTS: dict[str, np.ndarray] = {
    "empty": np.zeros((7, 7), dtype=int),
    "spiral": tiny_walls.copy(),
    "spiral_harder": tiny_walls.copy(),
}

# classic 15Ã21 maze with corridors 
# Row 7 is the portal row (can wrap around left<->right)
classic_board = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Portal row!
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=int)

LAYOUTS["classic"] = classic_board

# Portal row for classic layout
PORTAL_ROW = 7

# colour table (indices used in board array)
# 0 empty, 1 wall, 2 pac-man, 3 red ghost, 4 pellet, 5 blue ghost
COLOR_TABLE = np.array(
    [[0, 0, 0], [80, 80, 80], [255, 255, 0],
     [255, 0, 0], [0, 255, 0], [0, 128, 255]], dtype=np.uint8)

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


# PacmanEnv 
class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # init
    def __init__(self, layout: str = "empty", reward_shaping: bool = True):
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(layout)
        self.layout_name = layout
        self.floor = LAYOUTS[layout]  # 0/1 grid
        self.h, self.w = self.floor.shape  # board dims
        self.img_h = self.h * PIXELS_PER_CELL
        self.img_w = self.w * PIXELS_PER_CELL
        
        # Enable/disable reward shaping
        self.reward_shaping = reward_shaping

        self._build_tiles()  # pixel-art sprites

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0, 255, shape=(self.img_h, self.img_w, 3), dtype=np.uint8
        )

        self.rng = np.random.default_rng()
        self.reset()

    # helpers
    def _legal_neighbours(self, x: int, y: int) -> List[Tuple[int, int]]:
        out = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            
            # Handle portal wraparound for classic layout
            if self.layout_name == "classic" and x == PORTAL_ROW:
                if ny < 0:
                    ny = self.w - 1  # Wrap to right side
                elif ny >= self.w:
                    ny = 0  # Wrap to left side
            
            if 0 <= nx < self.h and 0 <= ny < self.w and self.floor[nx, ny] == 0:
                out.append((nx, ny))
        return out

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Manhattan distance, with portal-aware horizontal distance on the classic tunnel row."""
        x1, y1 = pos1
        x2, y2 = pos2

        # On classic portal row, horizontal distance should wrap
        if self.layout_name == "classic" and x1 == x2 == PORTAL_ROW:
            dy = abs(y1 - y2)
            # wrapped distance around the tunnel
            dy = min(dy, self.w - dy)
            return dy  # dx = 0 here because x1 == x2

        # Everywhere else: normal Manhattan
        return abs(x1 - x2) + abs(y1 - y2)


    def _min_pellet_distance(self, pos: Tuple[int, int]) -> int:
        """Get minimum Manhattan distance to any pellet."""
        if not self.pellets:
            return 0
        return min(self._manhattan_distance(pos, p) for p in self.pellets)

    def _min_ghost_distance(self, pos: Tuple[int, int]) -> int:
        """Get minimum Manhattan distance to any ghost."""
        if not self.ghost_pos:
            return float('inf')
        return min(self._manhattan_distance(pos, g) for g in self.ghost_pos)

    # reset
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.layout_name == "empty":
            self.pac_pos = (0, 0)
            self.ghost_pos = [(6, 6)]
            self.pellets = [(0, 6), (6, 0)]

        elif self.layout_name == "spiral":
            self.pac_pos = (1, 1)
            self.ghost_pos = [(5, 5)]
            self.pellets = [(1, 5), (5, 1)]

        elif self.layout_name == "spiral_harder":
            self.pac_pos = (1, 1)
            self.ghost_pos = [(3, 3)]
            self.pellets = [(1, 5), (5, 1)]

        else:  # classic
            self.pac_pos = (13, 10)  # lower middle corridor
            self.ghost_pos = [(7, 9), (7, 11)]  # red, blue - near center
            # pellets everywhere except walls and starting positions
            self.pellets = [(i, j) for i in range(self.h) for j in range(self.w)
                            if self.floor[i, j] == 0 and
                            (i, j) not in self.ghost_pos + [self.pac_pos]]

        self.first_ghost_move = True  # only used on "empty"
        self.ghost_dir: List[Tuple[int, int] | None] = [None] * len(self.ghost_pos)
        self.step_count = 0
        self.initial_pellet_count = len(self.pellets)
        
        return self._render_board(), {}

    # step
    def step(self, action: int):
        self.step_count += 1
        
        # Store old position for reward shaping
        old_pac_pos = self.pac_pos
        old_pellet_dist = self._min_pellet_distance(old_pac_pos) if self.reward_shaping else 0
        old_ghost_dist = self._min_ghost_distance(old_pac_pos) if self.reward_shaping else 0

        # Move Pac-Man
        px, py = self.pac_pos
        new_px, new_py = px, py
        
        if action == 0:    # UP
            new_px = px - 1
        elif action == 1:  # DOWN
            new_px = px + 1
        elif action == 2:  # LEFT
            new_py = py - 1
        elif action == 3:  # RIGHT
            new_py = py + 1
        
        # PPPPPPPPPPPPPPP PORTAL MECHANICS PPPPPPPPPPPPPPP
        # Handle portal wraparound for classic layout on row 7
        if self.layout_name == "classic" and px == PORTAL_ROW:
            if new_py < 0:
                new_py = self.w - 1  # Wrap to right edge
            elif new_py >= self.w:
                new_py = 0  # Wrap to left edge
        else:
            # Normal boundary clamping
            new_px = max(0, min(new_px, self.h - 1))
            new_py = max(0, min(new_py, self.w - 1))
        
        # Check for wall collision (revert if hitting wall)
        if self.floor[new_px, new_py] == 1:
            new_px, new_py = px, py
        
        self.pac_pos = (new_px, new_py)

        # Base step penalty
        reward = -0.1
        terminated = False

        # PPPPPPPPPPPPPPP REWARD SHAPING PPPPPPPPPPPPPPP
        if self.reward_shaping:
            new_pellet_dist = self._min_pellet_distance(self.pac_pos)
            new_ghost_dist = self._min_ghost_distance(self.pac_pos)
            
            # Reward for moving toward pellets
            if self.pellets:
                pellet_progress = old_pellet_dist - new_pellet_dist
                reward += 0.3 * pellet_progress  # +0.3 for getting closer
            
            # Penalty for being close to ghosts (danger zone)
            if new_ghost_dist <= 3:
                # Stronger penalty when closer
                danger_penalty = 0.8 * (4 - new_ghost_dist)
                reward -= danger_penalty
            
            # Bonus for increasing distance from ghosts when in danger
            if old_ghost_dist <= 4 and new_ghost_dist > old_ghost_dist:
                reward += 0.3  # Reward for escaping
            
            # Penalty for staying still (hitting wall)
            if self.pac_pos == old_pac_pos:
                reward -= 0.2

        # PPPPPPPPPPPPPPP CHECK 1: Collision after Pac-Man moves PPPPPPPPPPPPPPP
        if self.pac_pos in self.ghost_pos:
            reward -= 50
            terminated = True
            return self._render_board(), reward, terminated, False, {}

        # PPPPPPPPPPPPPPP PELLET COLLECTION PPPPPPPPPPPPPPP
        if self.pac_pos in self.pellets:
            self.pellets.remove(self.pac_pos)
            reward += 10
            
            # Bonus for collecting all pellets (WIN)
            if not self.pellets:
                reward += 100  # Increased win bonus
                # Additional bonus for efficiency (fewer steps = better)
                if self.reward_shaping:
                    efficiency_bonus = max(0, 50 - self.step_count * 0.05)
                    reward += efficiency_bonus
                terminated = True

        # PPPPPPPPPPPPPPP MOVE GHOSTS PPPPPPPPPPPPPPP
        if not terminated:
            for g_idx, (gx, gy) in enumerate(self.ghost_pos):
                if self.layout_name == "empty":
                    # Empty layout: simple chase with randomness
                    if self.first_ghost_move:
                        neighbors = self._legal_neighbours(gx, gy)
                        if neighbors:
                            gx, gy = self.rng.choice(neighbors)
                        self.first_ghost_move = False
                    else:
                        dx, dy = new_px - gx, new_py - gy
                        if self.rng.random() < 0.7:
                            if abs(dx) > abs(dy):
                                gx += int(math.copysign(1, dx))
                            elif dy:
                                gy += int(math.copysign(1, dy))
                        else:
                            gx += self.rng.choice([-1, 0, 1])
                            gy += self.rng.choice([-1, 0, 1])
                        gx = int(np.clip(gx, 0, self.h - 1))
                        gy = int(np.clip(gy, 0, self.w - 1))
                        if self.floor[gx, gy]:
                            gx, gy = self.ghost_pos[g_idx]

                else:  # Corridor-following ghost (spiral, spiral_harder, classic)
                    dir_ = self.ghost_dir[g_idx]
                    if dir_ is None:
                        neighbors = self._legal_neighbours(gx, gy)
                        if neighbors:
                            next_pos = self.rng.choice(neighbors)
                            gx, gy = next_pos
                            self.ghost_dir[g_idx] = (gx - self.ghost_pos[g_idx][0],
                                                      gy - self.ghost_pos[g_idx][1])
                    else:
                        dx, dy = dir_
                        nx, ny = gx + dx, gy + dy
                        
                        # Handle ghost portal wraparound
                        if self.layout_name == "classic" and gx == PORTAL_ROW:
                            if ny < 0:
                                ny = self.w - 1
                            elif ny >= self.w:
                                ny = 0
                        
                        if 0 <= nx < self.h and 0 <= ny < self.w and self.floor[nx, ny] == 0:
                            legal = self._legal_neighbours(gx, gy)
                            corridor = len(legal) == 2 and (nx, ny) in legal
                            if corridor:
                                gx, gy = nx, ny
                            else:
                                candidates = [(lx - gx, ly - gy) for lx, ly in legal]
                                rev = (-dx, -dy)
                                if len(candidates) > 1 and rev in candidates:
                                    candidates.remove(rev)
                                if candidates:
                                    dx, dy = self.rng.choice(candidates)
                                    self.ghost_dir[g_idx] = (dx, dy)
                                    gx, gy = gx + dx, gy + dy
                        else:
                            legal = self._legal_neighbours(gx, gy)
                            if legal:
                                deltas = [(lx - gx, ly - gy) for lx, ly in legal]
                                dx, dy = self.rng.choice(deltas)
                                self.ghost_dir[g_idx] = (dx, dy)
                                gx, gy = gx + dx, gy + dy

                self.ghost_pos[g_idx] = (gx, gy)

            # PPPPPPPPPPPPPPP CHECK 2: Collision after ghosts move PPPPPPPPPPPPPPP
            if self.pac_pos in self.ghost_pos:
                reward -= 50
                terminated = True

        return self._render_board(), reward, terminated, False, {}

    # tile builder
    def _build_tiles(self):
        """Build 12Ã12 pixel-art tiles."""
        s = PIXELS_PER_CELL
        cx = (s - 1) / 2
        yy, xx = np.mgrid[0:s, 0:s]
        tiles = np.zeros((6, s, s, 3), dtype=np.uint8)

        # wall
        tiles[1, :, :, :] = [80, 80, 80]

        # pac-man
        circle = (xx - cx) ** 2 + (yy - cx) ** 2 <= (s * 0.48) ** 2
        mouth = np.abs(np.arctan2(yy - cx, xx - cx)) < np.pi / 5
        tiles[2, circle & ~mouth] = [255, 255, 0]

        # helper to build ghost (red then blue)
        def ghost_tile(rgb):
            g = np.zeros((s, s), bool)
            g |= (yy - 4) ** 2 + (xx - cx) ** 2 <= 25  # round head
            g |= yy >= 4  # body
            for col in range(0, s, 4):  # legs
                g[s - 1, col + 2:col + 4] = False
            tile = np.zeros((s, s, 3), np.uint8)
            tile[g] = rgb
            for ex in (int(s * 0.28), int(s * 0.58)):
                ey = int(s * 0.33)
                tile[ey:ey + 3, ex:ex + 2] = [255, 255, 255]  # whites
                tile[ey + 1, ex + 1] = [0, 0, 0]  # pupil
            return tile

        tiles[3] = ghost_tile([255, 0, 0])  # red
        tiles[5] = ghost_tile([0, 128, 255])  # blue

        # pellet
        p0 = int(cx) - 1
        tiles[4, p0:p0 + 2, p0:p0 + 2] = [0, 255, 0]

        self.tiles = tiles

    # rendering
    def _render_board(self) -> np.ndarray:
        board = np.zeros((self.h, self.w), np.uint8)
        board[self.floor == 1] = 1
        for (x, y) in self.pellets:
            board[x, y] = 4
        for idx, (gx, gy) in enumerate(self.ghost_pos):
            board[gx, gy] = 3 if idx == 0 else 5
        px, py = self.pac_pos
        board[px, py] = 2

        img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        s = PIXELS_PER_CELL
        for i in range(self.h):
            for j in range(self.w):
                img[i * s:(i + 1) * s, j * s:(j + 1) * s] = self.tiles[board[i, j]]
        return img

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self._render_board()
        plt.imshow(self._render_board())
        plt.axis("off")
        plt.show()


# TEST PORTAL MECHANICS
if __name__ == "__main__":
    print("Testing portal mechanics...")
    env = PacmanEnv("classic")
    
    # Manually set pac-man to portal row
    env.pac_pos = (7, 0)  # Left edge of portal row
    print(f"Starting position: {env.pac_pos}")
    
    # Try moving left (should wrap to right side)
    env.step(2)  # LEFT
    print(f"After LEFT from (7,0): {env.pac_pos}")  # Should be (7, 20)
    
    # Reset to right edge
    env.pac_pos = (7, 20)
    print(f"Position: {env.pac_pos}")
    
    # Try moving right (should wrap to left side)
    env.step(3)  # RIGHT
    print(f"After RIGHT from (7,20): {env.pac_pos}")  # Should be (7, 0)
    
    print("\nPortal test complete!")
    
    # Count pellets on each side
    env.reset()
    left_pellets = sum(1 for p in env.pellets if p[1] < 10)
    right_pellets = sum(1 for p in env.pellets if p[1] >= 10)
    print(f"Pellets on left side: {left_pellets}")
    print(f"Pellets on right side: {right_pellets}")
    print(f"Total pellets: {len(env.pellets)}")