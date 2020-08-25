from abc import ABC, abstractmethod

# Shortcuts for atari games. Other gym games are handled differently
GYM_SHORTCUTS = {
    'GYMbrkt': 'GYMbreakout',
    'GYMp': 'GYMpong',
    'GYMsi': 'GYMspace_invaders',
    'GYMcc': 'GYMchopper_command',
    'GYMbx': 'GYMboxing',
    'GYMsq': 'GYMseaquest',
    'GYMat': 'GYMatlantis',
    'GYMkfm': 'GYMkung_fu_master',
}

ALL_ENVS = ['MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-SimpleCrossingS9N2-v0',
    'MiniGrid-SimpleCrossingS9N3-v0',
    'MiniGrid-SimpleCrossingS11N5-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'CartPole-v0'] + list(GYM_SHORTCUTS.keys()) + list(GYM_SHORTCUTS.values())

def get_env(name, seed, **kwargs):
    if name.startswith('MiniGrid'):
        from environments.minigrid.envspec import get_env as getter
    elif name == 'CartPole-v0':
        from environments.cartpole.envspec import get_env as getter
    elif name[:3] == 'GYM':
        name = GYM_SHORTCUTS.get(name, name)
        name = name[3:]
        from environments.gym_atari.envspec import get_env as getter
    else:
        print(name, "environment not found, try another from:\n" + "\n".join(ALL_ENVS))
        exit()

    env = getter(name, seed, **kwargs)
    return env

class AbstractEnv(ABC):
    """ Defines all properties environments are assumed to have """
    def __init__(self, do_nothing, actions):
        self.do_nothing = do_nothing
        self.actions = actions

    @abstractmethod
    def step(self, action):
        """ Should follow OpenAI gym style returns:
        return state, reward, done, info """
        pass

    @abstractmethod
    def reset(self):
        """ Should follow OpenAI gym style returns:
        return state """
        pass

    @abstractmethod
    def abst(self, state):
        """ Abstraction used for mutating states. Should return hashable
        identifier for state. """
        pass

    @abstractmethod
    def close(self):
        """ Stop environment """
        pass

    def get_RGB(self, env, state, action, mut, score):
        """ Returns frames used for visualisation.see_env.run_and_save
        Not necessary for running count/score/interpol loop. """
        raise NotImplementedError

### Env tools ###
def run_env_with(env, pol, func):
    args = {
        'env': env,
        'pol': pol,
        'states': [env.reset()],
        'acts': [],
        'rews': [],
        'done': False,
        'step': 0
    }
    func(args)

    while not args['done']:
        args['acts'].append(pol(args['states'], args['acts'], args['rews']))

        s, r, d, _ = env.step(args['acts'][-1])

        args['states'].append(s)
        args['rews'].append(r)
        args['done'] = d
        args['step'] += 1

        func(args)

class Get_stats():
    def __init__(self, cond):
        self.cond = cond
        self.tot_r = 0
        self.n_mut = 0
        self.steps = 0
        self.pss = self.cond([], [], [])

    def __call__(self, args):
        self.tot_r = sum(args['rews'])
        self.n_mut += 1 if args['pol'].was_last_mut() and len(args['acts']) > 0 else 0
        self.steps = args['step']
        self.pss = self.cond(args['states'], args['acts'], args['rews'])

    def get_stats(self, reset=False):
        res = self.tot_r, self.pss, self.n_mut, self.steps
        if reset:
            self.reset()
        return res

    def reset(self):
        self.__init__(self.cond)
