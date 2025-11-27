# frame_stack.py
import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, env, k=4):
        self.env = env
        self.k = k
        self.frames = deque(maxlen=k)
        obs_shape = env.observation_space.shape
        self.shape = (obs_shape[0], obs_shape[1], obs_shape[2] * k)
        self.observation_space = env.observation_space
        self.action_space = env.action_space


    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

