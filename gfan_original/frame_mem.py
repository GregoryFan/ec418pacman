from collections import deque
import numpy as np
from gymnasium import spaces
import cv2

class FrameStack:
    def __init__(self, env, k=4):
        #Pacman Env
        self.env = env
        #Number of stacked frames
        self.k = k 
        self.frames = deque(maxlen=k)
        self.resize = (84, 84) 

    #Reset Environment
    def reset(self):
        obs, info = self.env.reset()
        obs = cv2.resize(obs, self.resize, interpolation=cv2.INTER_AREA)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = cv2.resize(obs, self.resize, interpolation=cv2.INTER_AREA)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    #Get Observations
    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

    @property
    def observation_space(self):
        old_space = self.env.observation_space
        w, h = self.resize 
        c = old_space.shape[2]
        return spaces.Box(low=old_space.low.min(), 
                          high=old_space.high.max(), 
                          shape=(h, w, c * self.k), 
                          dtype=old_space.dtype)
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)
    
    @property
    def pellets(self):
        return self.env.pellets
    
    @property
    def pac_pos(self):
        return self.env.pac_pos
    
    @property
    def ghost_pos(self):
        return self.env.ghost_pos