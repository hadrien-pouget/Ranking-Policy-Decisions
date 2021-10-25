import os
import argparse

from polrank.elements.envs import atari_games_clean

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, choices=atari_games_clean + ['Minigrid'] + ['CartPole'])
parser.add_argument('--N', type=int, help="Number of times to repeat experiment", default=1)
parser.add_argument('--rand', action='store_true', help="Use random action as default?")
args = parser.parse_args()
env = args.env
N = args.N
pol_d = 'random' if args.rand else 'lazy'

# Dict of commands
tasks = {
    # Minigrid
    'Minigrid': 'python polrank --env_name MiniGrid-SimpleCrossingS9N1-v0 --pol_name minigrid_good --pol_d_name {} --cond_name score0.8 -nr 5000 -nt 100 -ni 200 -mu 0.2',
    # CartPole
    'CartPole': 'python polrank --env_name CartPole-v0 --pol_name CartPole_good --pol_d_name {} --cond_name score200 -nr 5000 -nt 100 -ni 100 -mu 0.4',
    # Atari Games
    'Atari':  'python polrank --env_name UBER{} --pol_name UBER{} --pol_d_name {} --cond_name score_auto -nr 1000 -nt 50 -ni -1 -mu 0.2 -a -1 --max_steps 600',
}

if env in atari_games_clean:
    command = tasks['Atari'].format(env, env, pol_d)
else:
    command = tasks[env].format(pol_d)

# Run the command and save to a good file name
os.system("{} -fl {}_0".format(command, env))

# Run subsequent repeats of the command using the same config as the first
# This is especially useful for score_auto, which will set a balanced condition
# on-the-fly in the first run.
for i in range(1, N):
    os.system("{} -fl {}_{} -ll {}_0".format(command, env, i, env))
