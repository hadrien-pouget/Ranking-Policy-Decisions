import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('task', type=int)
task = parser.parse_args().task

# Dict for legibility
tasks = {
    # Minigrid
    0: 'python polexp --env_name MiniGrid-SimpleCrossingS9N1-v0 --pol_name minigrid_good --pol_d_name lazy --cond_name score0.8 -ni 200 -nr 5000 -nt 100 -mu 0.2',
    # CartPole
    1: 'python polexp --env_name CartPole-v0 --pol_name CartPole_good --cond_name score200 -nr 5000 -nt 100 -mu 0.4 -ni 100',
    # Atari Games
    2: 'python polexp --env_name GYMbrkt --pol_name GYMbrkt --pol_d_name lazy --cond_name score19   -nr 1000 -nt 50 -ni -1   -mu 0.2 -a 2  --max_steps 600',
    3: 'python polexp --env_name GYMbrkt --pol_name GYMbrkt --pol_d_name lazy --cond_name score19   -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    4: 'python polexp --env_name GYMsi   --pol_name GYMsi   --pol_d_name lazy --cond_name score550  -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    5: 'python polexp --env_name GYMp    --pol_name GYMp    --pol_d_name lazy --cond_name score0    -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    6: 'python polexp --env_name GYMcc   --pol_name GYMcc   --pol_d_name lazy --cond_name score3100 -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    7: 'python polexp --env_name GYMbx   --pol_name GYMbx   --pol_d_name lazy --cond_name score32   -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    8: 'python polexp --env_name GYMsq   --pol_name GYMsq   --pol_d_name lazy --cond_name score750  -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    9: 'python polexp --env_name GYMat   --pol_name GYMat   --pol_d_name lazy --cond_name score3000 -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
    10: 'python polexp --env_name GYMkfm  --pol_name GYMkfm  --pol_d_name lazy --cond_name score3000 -nr 1000 -nt 50 -ni -1   -mu 0.2 -a -1 --max_steps 600',
}

t = tasks[task]
os.system(t)
