import os
from datetime import datetime
import random
import argparse

import torch

from scoring import ALL_SCORE_TYPES
from utils.logging import RESULTS_LOC
from elements.envs import ALL_ENVS
from elements.policies import ALL_POLS

def get_date_num():
    dirs = os.listdir('./' + RESULTS_LOC)
    date = datetime.strftime(datetime.now(), "%m-%d")
    num = 0
    floc = date + '_' + str(num)
    while floc in dirs:
        num += 1
        floc = date + '_' + str(num)
    return floc

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make policy interpolations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    subparsers = parser.add_subparsers(help='Do something else')
    parser.set_defaults(task=None)

    # General
    parser.add_argument('--env_name', '-e', default='MiniGrid-SimpleCrossingS9N1-v0',
        choices=ALL_ENVS)
    parser.add_argument('--env_seed', '-es', type=int, default=1234,
        help="-1 for random seed.")
    parser.add_argument('--max_steps', type=int, default=-1,
        help="Default (-1) means no max. Not implemented for all envs.")
    parser.add_argument('--pol_name', '-p', default='minigrid_good',
        choices=ALL_POLS)
    parser.add_argument('--pol_d_name', '-pd', default='lazy',
        choices=ALL_POLS)
    parser.add_argument('--cond_name', '-cd', default='score0.85', 
        help="What condition is being tested? Look in element/conditions for options.")
    parser.add_argument('--no_det', action='store_true',
        help="By default, all randomness is seeded by env_seed. Use this to allow for \
        randomness. Environment randomness is still seeded by env_seed.")

    # Saving
    def_fileloc = get_date_num()
    parser.add_argument('--fileloc', '-fl', default=def_fileloc,
        help="Where to save.")
    parser.add_argument('--load_loc', '-ll', default=None, 
        help="Load config from this location, overwriting arguments given at command line.")

    # Counting
    parser.add_argument('--n_runs', '-nr', type=int, default=500,
        help="Number of runs through environment when building test set.")
    parser.add_argument('--mut_prob', '-mu', type=float, default=0.3,
        help="Probably of using default policy in each step.")

    # Scoring
    parser.add_argument('--score_types', '-st', nargs='+',
        choices=ALL_SCORE_TYPES)

    # Interpolating
    parser.add_argument('--n_inc', '-ni', type=int, default=-1,
        help="Number of states to add in at each step of interpolation. -1 does \
            number of states for 12 interpolation steps.")
    parser.add_argument('--n_test', '-nt', type=int, default=50,
        help="Number of executions to test policies during interpolation.")

    # Environment processing functions
    parser.add_argument('--abst_type', '-a', type=int, default=-1)
    parser.add_argument('--vis_type', type=int, default=-1)

    # Redoing results
    parser.add_argument('--redo_all', action='store_true')
    parser.add_argument('--more_count', type=int)
    parser.add_argument('--more_scoretypes', nargs='+',
        choices=ALL_SCORE_TYPES)
    parser.add_argument('--redo_interpol', '-ri', type=int, nargs=2,
        help="Give new values for n_inc and n_test. --redo_interpol n_inc n_test. Put -1 \
            for no change in that parameter, put -1 in both to simply redo interpolation.")
    parser.add_argument('--skip_load', action='store_true',
        help="Skips loading everything but results. Use if you only want to load and work with pre-existing results.")

    # Torch
    def_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=def_device, choices=['cuda', 'cpu'])

    ### Visualisation subparser
    vis = subparsers.add_parser('vis')
    vis.add_argument('--not_mut', '-nm', type=int, default='all', 
        help="Int - don't use the default action in the top 'n' states")
    vis.add_argument('--ranking', '-r', choices=ALL_SCORE_TYPES)
    vis.add_argument('--vis_seed', '-vs', type=int, default=None,
        help="If during visualisation you want to run the environment with a different seed, \
        use this to choose it.")
    vis.add_argument('--get_fail', action='store_true')
    vis.add_argument('--get_succ', action='store_true')
    vis.set_defaults(task='vis')

    args = parser.parse_args()
    if args.score_types is None:
        args.score_types = ['ochiai', 'tarantula', 'zoltar', 'wongII', 'freqVis', 'rand']
    else:
        args.score_types = args.score_types.split(' ')

    if args.more_scoretypes is not None:
        args.more_scoretypes = args.more_scoretypes.split(' ')

    args.env_seed = random.randint(1000, 9999) if args.env_seed == -1 else args.env_seed
    if args.task == 'vis':
        args.vis_seed = random.randint(1000, 9999) if args.vis_seed == -1 else args.vis_seed
    else:
        args.vis_seed = None
    # use_seed is the seed that will actually be used to seed the environment.
    # If vis_seed is provided, will be used over the env_seed.
    # Only the env_seed is saved in the config.
    args.use_seed = args.vis_seed if args.vis_seed is not None else args.env_seed

    return args
