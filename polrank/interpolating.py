import time

import numpy as np
from tqdm import tqdm

from elements.envs import run_env_with, Get_stats
from elements.policies import MixedPol, RandomRankingPol
from utils.timing import sec_to_str

INTERPOL_COLS = [
    "Number of Active States in Policy",
    "Reward Average",
    "Reward Variance",
    "Fraction Passing Executions",
    "Fraction Actions Taken Inactively",
    "Number of Active Actions Taken",
]

def interpolate(logger, score_types=None):
    """ Set up interpolation, with score types either
    from logger or overridden by provided ones """
    config = logger.config
    score_types = score_types if score_types is not None else config['score_types']
    update_logs = logger.update_logs
    return interpolate_policies(
        config['env'],
        config['pol'],
        config['pol_d'],
        config['cond'],
        logger.data['scores'][0],
        score_types,
        config['n_inc'],
        config['n_test'],
        update_logs,
    )

def interpolate_policies(env, pol, pol_d, cond, rankings, score_types, n_inc, n_test, update_logs):
    """ Given two policies, interpolates between them
    and saves outcomes for each interpolation. In last step does
    completely unmutated policy, and puts it at n_inc past the last index """
    start = time.time()

    results = {}
    for st in score_types:
        ranking = rankings[st]
        state_ranking = [s for s, sc in ranking]

        inds, avgs, vrs, chks, mut_ps, n_muts = [], [], [], [], [], []
        print("\nBeginning interpolation for ranking with score type:", st)
        # This goes through whole ranking, and then does complete policy
        for i in tqdm(list(range(0, len(state_ranking), n_inc)) + [len(state_ranking), -1]):
            if st == 'rand':
                mpol = RandomRankingPol(pol, pol_d, state_ranking, i, abst=env.abst)
            else:
                not_mut = state_ranking[:i] if i >= 0 else 'all'
                mpol = MixedPol(pol, pol_d, not_mut, abst=env.abst)

            tot_rs, passes, mut_props, not_muts = test_pol(env, mpol, cond, n_test, rand=st=='rand')

            # Results for complete policy are kept under index '-1', at end of list
            inds.append(i)
            avgs.append(np.mean(tot_rs))
            vrs.append(np.var(tot_rs))
            chks.append(np.mean(passes))
            mut_ps.append(np.mean(mut_props))
            n_muts.append(np.mean(not_muts))
            # print("Done interpolation with {}/{} mutations".format(i, len(state_ranking)),
            #     end='\r' if i < len(state_ranking) else '\n')

        results[st] = inds, avgs, vrs, chks, mut_ps, n_muts

    end = time.time()
    log = {
        'interpol_time': sec_to_str(end - start)
    }
    update_logs(log)

    return results

def test_pol(env, pol, cond, n_test, rand=False):
    """ Test a polciy n_test times, keeping track
    of outcomes """
    tot_rs, passes, not_muts, mut_props = [], [], [], []
    stats = Get_stats(cond)

    for _ in range(n_test):
        if rand:
            pol.shuffle_rank()
        run_env_with(env, pol, stats)
        tot_r, pss, mut_n, steps = stats.get_stats(reset=True)
        tot_rs.append(tot_r)
        passes.append(pss)
        mut_props.append(mut_n/steps)
        not_muts.append(steps-mut_n)

    return tot_rs, passes, mut_props, not_muts
