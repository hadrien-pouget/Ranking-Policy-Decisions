from abc import ABC, abstractmethod

# From https://github.com/openai/gym/blob/2d247dc93a8c98360ebeb6a3807a9b3d945424ee/gym/envs/__init__.py 
atari_games = ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']
atari_games_clean = [''.join([g.capitalize() for g in game.split('_')]) for game in atari_games]
# ['Adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 
# 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival', 
# 'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk', 'ElevatorAction', 
# 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond', 
# 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 
# 'Phoenix', 'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 
# 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 
# 'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']

ALL_ENVS = ['MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-SimpleCrossingS9N2-v0',
    'MiniGrid-SimpleCrossingS9N3-v0',
    'MiniGrid-SimpleCrossingS11N5-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'CartPole-v0'] + \
    ['UBER' + game for game in atari_games_clean] + \
    ['GYM'  + game for game in atari_games]

def get_env(name, seed, **kwargs):
    if name.startswith('MiniGrid'):
        from environments.minigrid.envspec import get_env as getter
    elif name == 'CartPole-v0':
        from environments.cartpole.envspec import get_env as getter
    elif name[:3] == 'GYM':
        name = name[3:]
        from environments.gym_atari.envspec import get_env as getter
    elif name[:4] == 'UBER':
        name = name[4:]
        from environments.uber_gym.envspec import get_env as getter
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
