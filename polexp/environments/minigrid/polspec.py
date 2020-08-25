import os

from elements.policies import AbstractPol
import environments.minigrid.utils as mg_utils

def get_pol(name, env, device, **kwargs):
    model_dir = os.path.join('polexp', 'environments', 'minigrid', 'storage', name[9:])
    agent = mg_utils.Agent(env.observation_space, env.action_space, model_dir, device, False, 1)
    return MinigridAgentPol(agent)

class MinigridAgentPol(AbstractPol):
    def __call__(self, states, actions, rews):
        return self.pol.get_action(states[-1])
