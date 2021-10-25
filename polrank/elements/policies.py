from abc import ABC, abstractmethod
import random

from elements.envs import atari_games
from elements.envs import atari_games_clean

ALL_POLS = ['lazy', 'donoth', 'random', 'minigrid_good', 'minigrid_lava', 'CartPole_good', 'CartPole_bad']  + \
    ['UBER' + game for game in atari_games_clean] + \
    ['GYM'  + game for game in atari_games]

def get_pol(name, env, device, **kwargs):
    if name == 'lazy':
        return LazyPol(**kwargs)
    if name == 'donoth':
        return DoNothPol(do_nothing=env.do_nothing)
    if name == 'random':
        return RandomPol(actions=env.actions)

    if name[:8] == 'minigrid':
        from environments.minigrid.polspec import get_pol as getter
    elif name[:3] == 'GYM':
        name = name[3:]
        from environments.gym_atari.polspec import get_pol as getter
    elif name[:4] == 'UBER':
        name = name[4:]
        from environments.uber_gym.polspec import get_pol as getter
    elif name[:8] == 'CartPole':
        from environments.cartpole.polspec import get_pol as getter
    else:
        print(name, "policy not found, try another from:\n", "\n".join(ALL_POLS))
        exit()

    pol = getter(name, env, device, **kwargs)
    return pol

class AbstractPol(ABC):
    def __init__(self, pol):
        self.pol = pol

    @abstractmethod
    def __call__(self, states, actions, rews):
        pass

class LazyPol(AbstractPol):
    def __init__(self, default=0):
        super().__init__(None)
        self.default = default

    def __call__(self, states, actions, rews):
        if len(actions) > 0:
            return actions[-1]
        else:
            return self.default

class DoNothPol(AbstractPol):
    def __init__(self, do_nothing=None):
        super().__init__(None)
        if do_nothing is None:
            raise NotImplementedError('No do_nothing provided!')
        self.do_nothing = do_nothing

    def __call__(self, states, actions, rews):
        return self.do_nothing

class RandomPol(AbstractPol):
    def __init__(self, actions=None):
        super().__init__(None)
        if actions is None:
            raise NotImplementedError('No action space provided for random sampling!')
        self.actions = actions

    def __call__(self, states, actions, rews):
        return random.choice(self.actions)

class MixedPol(AbstractPol):
    """ Mix two policies by using pol_d everywhere except in
    not_mut states, in which case use pol. abst is abstraction
    function used for not_mut. """

    def __init__(self, pol, pol_d, not_mut, abst=None):
        super().__init__(pol)
        self.pol_d = pol_d
        self.not_mut = not_mut
        self.abst = abst if abst is not None else (lambda x: x)
        self.was_mut = False

    def __call__(self, states, actions, rews):
        if self.not_mut == 'all' or self.abst(states[-1]) in self.not_mut:
            self.was_mut = False
            return self.pol(states, actions, rews)
        else:
            self.was_mut = True
            return self.pol_d(states, actions, rews)

    def was_last_mut(self):
        return self.was_mut

class RandomRankingPol(AbstractPol):
    """ Policy which takes all states when initialized,
    ranks them randomly, and chooses top n as not
    mutated.
    self.shuffle_rank re-shuffles ranking
    self.set_n sets number of unmutated states """

    def __init__(self, pol, pol_d, states, init_not_mut, abst=None):
        super().__init__(pol)
        self.pol_d = pol_d

        self.states = states
        self.n = init_not_mut
        self.not_mut = []
        self.shuffle_rank()

        self.abst = abst if abst is not None else (lambda x: x)
        self.was_mut = False

    def shuffle_rank(self):
        random.shuffle(self.states)
        self.set_n(self.n)

    def set_n(self, n):
        if n == -1:
            self.not_mut = 'all'
        else:
            n = min(len(self.states), n)
            self.not_mut = self.states[:n]
        self.n = n

    def __call__(self, states, actions, rews):
        if self.not_mut == 'all' or self.abst(states[-1]) in self.not_mut:
            self.was_mut = False
            return self.pol(states, actions, rews)
        else:
            self.was_mut = True
            return self.pol_d(states, actions, rews)

    def was_last_mut(self):
        return self.was_mut
