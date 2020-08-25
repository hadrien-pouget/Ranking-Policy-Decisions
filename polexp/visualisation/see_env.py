import os
import json

import imageio

from elements.envs import run_env_with, Get_stats
from elements.policies import MixedPol

def run_and_save(logger):
    config = logger.config
    results = env_with_frame_proc(
        config['env'],
        config['pol'],
        config['pol_d'],
        config['cond'],
        logger.data['scores'][0][config['ranking']],
        config['not_mut'],
        config['get_fail'],
        config['get_succ'],
    )
    save_vis(logger.fileloc, results, config['ranking'], config['not_mut'], config['use_seed'])

def env_with_frame_proc(env, pol, pol_d, cond, ranking, n_not_mut, get_failure=False, get_success=False):
    # Get mixed policy
    if n_not_mut >= 0:
        ranking.sort(key=lambda tup: tup[1], reverse=True)
        not_mut = ranking[:n_not_mut]
        not_mut = [s for s, sc in not_mut]
    else:
        not_mut = 'all'
    pol = MixedPol(pol, pol_d, not_mut, abst=env.abst)

    # Run through environment
    stats = Get_stats_frames(cond, dict(ranking), env.get_RGB)
    assert not (get_failure and get_success), "If you want a successful run and a failed run, do them separately!"
    cont = True
    while cont:
        stats.reset()
        run_env_with(env, pol, stats)
        succ = stats.get_stats()[1]
        cont = False or (get_failure and succ) or (get_success and not succ)
    env.close()
    return stats.get_stats()

def save_vis(fileloc, results, score_type, n_not_mut, seed):
    seed = str(seed) if seed is not None else ''
    sloc = os.path.join(fileloc, get_gif_floc(fileloc, score_type, n_not_mut, seed))
    os.makedirs(sloc)

    for i, im in enumerate(results[4]):
        imageio.imsave(os.path.join(sloc, 'frame_{0:03d}.png'.format(i)), im)

    imageio.mimsave(
        os.path.join(sloc, '{}_{}_s{}.gif'.format(score_type, n_not_mut, seed)),
        results[4],
        fps=5)

    config = {
        'rew': results[0],
        'pass': results[1],
        'n_mut (during run)': results[2],
        'steps': results[3],
        'prop_mut': results[2] / results[3],
        'score_type': score_type,
        'n_not_mut (in all of policy)': n_not_mut,
        'seed': seed
    }

    js = json.dumps(config, sort_keys=True, indent=4, separators=(',', ':'))
    with open(os.path.join(sloc, 'config.json'), 'w') as f:
        f.write(js)

def get_gif_floc(fileloc, score_type, n_not_mut, seed=''):
    dirs = os.listdir(fileloc)
    num = 0
    floc = 'gif_{}_{}_s{}_{}'.format(score_type, n_not_mut, seed, num)
    while floc in dirs:
        num += 1
        floc = 'gif_{}_{}_s{}_{}'.format(score_type, n_not_mut, seed, num)
    return floc

class Get_stats_frames(Get_stats):
    def __init__(self, cond, score_lookup, get_RGB):
        super().__init__(cond)
        self.get_RGB = get_RGB
        self.frames = []
        self.score_lookup = score_lookup

    def __call__(self, args):
        super().__call__(args)
        env = args['env']
        state = args['states'][-1] if len(args['states']) > 0 else None
        action = args['pol'](args['states'], args['acts'], args['rews'])

        abs_state = env.abst(args['states'][-1])
        mut = abs_state not in args['pol'].not_mut and args['pol'].not_mut != 'all'
        score = self.score_lookup.get(abs_state, "Not scored")
        self.frames.append(self.get_RGB(env, state, action, mut, score))

    def get_stats(self, reset=False):
        res = super().get_stats()
        res = *res, self.frames
        if reset:
            self.reset()
        return res

    def reset(self):
        self.__init__(self.cond, self.score_lookup, self.get_RGB)
