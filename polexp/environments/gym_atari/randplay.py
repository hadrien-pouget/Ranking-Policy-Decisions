import sys
import random

import numpy as np

from env import Env

games = ['atlantis', 'breakout', 'pong', 'space_invaders', 'kung_fu_master', 'boxing', 'seaquest', 'chopper_command']

for game in games:
    env = Env(game, 1234, 'cuda', 600, 4, False)
    acts = env.action_space() - 1

    all_rews = []
    for i in range(100):
        env.reset()
        done = False
        rew = 0
        while not done:
            _, r, done, _ = env.step(random.randint(0, acts))
            rew += r
        all_rews.append(rew)
        print('Ep: ', i, end='\r')
    print(game, np.mean(all_rews))
