# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces
import tensorflow as tf
from pdb import set_trace as bb

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k, dual_state):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.dual_state = dual_state
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        if self.dual_state:
            ob, true_ob = self.env.reset()
            for _ in range(self.k): self.frames.append(ob)
            return self.observation(), true_ob
        else:
            ob = self.env.reset()
            for _ in range(self.k): self.frames.append(ob)
            return self.observation()

    def step(self, action):
        if self.dual_state:
            (ob, true_ob), reward, done, info = self.env.step(action)
            self.frames.append(ob)
            return (self.observation(), true_ob), reward, done, info
        else:
            ob, reward, done, info = self.env.step(action)
            self.frames.append(ob)
            return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env, dual_state, scale=(1/255.0)):
        gym.ObservationWrapper.__init__(self, env)
        self.dual_state = dual_state
        self.scale = scale

    def reset(self, **kwargs):
        if self.dual_state:
            observation, true_obs = self.env.reset(**kwargs)
            return self.observation(observation), true_obs
        else:
            observation = self.env.reset(**kwargs)
            return self.observation(observation)

    def step(self, action):
        if self.dual_state:
            (observation, true_obs), reward, done, info = self.env.step(action)
            return (self.observation(observation), true_obs), reward, done, info
        else:
            observation, reward, done, info = self.env.step(action)
            return self.observation(observation), reward, done, info

    def observation(self, obs):
    # careful! This undoes the memory optimization, use
    # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) * self.scale

class MaxStepsEnv(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)
        self.steps = None
        self.max_steps = max_steps

    def reset(self):
        self.steps = 0
        return self.env.reset()

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done or self.steps >= self.max_steps, info
