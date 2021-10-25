import numpy as np
import gym
import gym_minigrid

from elements.envs import AbstractEnv

def get_env(name, seed, **kwargs):
    env = MinigridEnv(name, seed)
    return env

class MinigridEnv(AbstractEnv):
    """ Uses the gym minigrid package. We treat the environment as
    being a single configuration of the environment, but configuration
    normally changes with each reset.

    actions:
    0 left
    1 right
    2 forward
    3-6 nothing """
    def __init__(self, name, seed):
        # self.seed = seed

        self.env = gym.make(name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.env.seed(seed)
        self.reset() # First reset changes layout for seed
        super().__init__(do_nothing=2, actions=list(range(self.action_space.n))) 

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        # Need to see each time, or layout changes
        # self.env.seed(self.seed)
        return self.env.reset()

    def abst(self, state):
        return str(state['image']) + ' ' + str(state['direction'])

    def close(self):
        self.env.close()

    def get_RGB(self, env, state, action, mut, scores):
        return env.env.render(mode='human')
