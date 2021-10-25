import os

from elements.policies import AbstractPol
import environments.minigrid.utils as mg_utils
from utils.download_weights import check_and_dwnld

download_links = {
    "good": "https://www.dropbox.com/s/b6hf9tmhpk5ms3q/minigrid_good.pt?dl=1",
    "lava": "https://www.dropbox.com/s/afued3ko9x13odo/minigrid_lava.pt?dl=1"
}

def get_pol(name, env, device, **kwargs):
    model_dir = os.path.join('polrank', 'environments', 'minigrid', 'storage', name[9:])
    check_and_dwnld(os.path.join(model_dir, 'status.pt'), download_links.get(name[9:], None))
    agent = mg_utils.Agent(env.observation_space, env.action_space, model_dir, device, False, 1)
    return MinigridAgentPol(agent)

class MinigridAgentPol(AbstractPol):
    def __call__(self, states, actions, rews):
        return self.pol.get_action(states[-1])
