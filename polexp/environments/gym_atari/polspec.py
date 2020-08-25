from elements.policies import AbstractPol
from environments.gym_atari.agent import get_agent

def get_pol(name, env, device, **kwargs):
    agent = get_agent(name, env, device)
    agent.eval()
    return RainbowAgent(agent)

class RainbowAgent(AbstractPol):
    def __init__(self, pol):
        super().__init__(pol)
        self.device = pol.device

    def __call__(self, states, actions, rews):
        return self.pol.act_e_greedy(states[-1][0])
