import os

import torch

from elements.policies import AbstractPol
from environments.cartpole.model import DuelDQNet
from utils.download_weights import check_and_dwnld

download_links = {
    "CartPole_good": "https://www.dropbox.com/s/4k1rxcz7bgm5pr9/CartPole_good.pth?dl=1",
    "CartPole_bad": "https://www.dropbox.com/s/8pw0c5p9dweg75b/CartPole_bad.pth?dl=1"
}

def get_pol(name, env, device, **kwargs):
    model_dir = os.path.join('polrank', 'environments', 'cartpole', name + '.pth')
    check_and_dwnld(model_dir, download_links.get(name, None))
    model = DuelDQNet(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    return CartPoleAgent(model, env, device)

class CartPoleAgent(AbstractPol):
    def __init__(self, pol, env, device):
        super().__init__(pol)
        self.env = env
        self.device = device

    def __call__(self, states, actions, rews):
        state = torch.tensor(states[-1], dtype=torch.float, device=self.device)
        state = state.unsqueeze(0)
        return self.pol.get_e_action(state, 0.01, self.env)
