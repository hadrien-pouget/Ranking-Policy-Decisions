import random
import time

import numpy as np

from utils.timing import sec_to_str

def count(logger, n_runs=None):
    """ Get counts for states, either as many as
    logger requests, or n_runs if overriding """
    config = logger.config
    n_runs = n_runs if n_runs is not None else config['n_runs']
    update_logs = logger.update_logs
    update_cond = logger.update_cond
    auto_cond = config['cond_name'][:5] == 'score' and config['cond_name'][5:] == '_auto'
    return get_counts(
        n_runs,
        config['env'],
        config['pol'],
        config['pol_d'],
        config['mut_prob'],
        config['cond'],
        update_logs,
        update_cond,
        auto_cond)

def get_counts(n_runs, env, pol, pol_d, mut_prob, cond, update_logs, update_cond, auto_cond=False):
    """ Get successes and failure counts for each state """
    all_start = time.time()

    counts = {}
    succs = 0
    tot_rews = []
    stepss = []
    print("\nBeginning counting")
    print("Done with 0/{} counting runs".format(n_runs), end='\r')
    for i in range(n_runs):
        start = time.time()

        mut_states, norm_states, succ, tot_rew, steps = run_env_with_muts(env, pol, pol_d, mut_prob, cond)
        succs += 1 if succ else 0
        tot_rews.append(tot_rew)
        stepss.append(steps)
        if auto_cond:
            update_flex_counts(counts, mut_states, norm_states, tot_rew)
        else:
            update_counts(counts, mut_states, norm_states, succ)

        end = time.time()
        est_time_left = sec_to_str(((end - all_start) / (i+1)) * (n_runs - i+1))

        # (general) passes, abstract states, total time < estimated time left | (episode) reward, steps, time
        print("Done with {}/{} counting runs, p:{} as:{} tt:{}<{} | r:{} s:{} t:{}".format(
            i+1, n_runs, succs, len(counts), sec_to_str(end-all_start), est_time_left, tot_rew, steps, sec_to_str(end-start)))

    all_end = time.time()

    if auto_cond:
        thresh = np.median(tot_rews)
        counts = flex_to_counts(counts, thresh)
        cond_name = 'score' + str(thresh)
        update_cond(cond_name)
        succs = sum(rew >= thresh for rew in tot_rews)

    logs = {
        'counting_abs_states': len(counts),
        # 'counting_rews': tot_rews,
        'counting_rews_mean': np.mean(tot_rews),
        # 'counting_steps': stepss,
        'counting_steps_mean': np.mean(stepss),
        'counting_succs': "{}/{}".format(succs, n_runs),
        'counting_time': sec_to_str(all_end - all_start),
        'counting_auto_cond?': auto_cond,
    }
    update_logs(logs)

    return counts

def run_env_with_muts(env, pol, pol_d, mut_prob, cond):
    """Run environment, making mutations to pol_d according to
    mut_prob, and return mutations, visited states, and condition"""
    mut_states = set()
    norm_states = set()

    s = env.reset()
    ss = env.abst(s)
    state_seq, action_seq, rew_seq = [s], [], []
    steps = 0
    done = False
    while not done:
        if ss in mut_states:
            a = pol_d(state_seq, action_seq, rew_seq)
        elif ss in norm_states:
            a = pol(state_seq, action_seq, rew_seq)
        elif random.random() > mut_prob:
            a = pol(state_seq, action_seq, rew_seq)
            norm_states.add(ss)
        else:
            a = pol_d(state_seq, action_seq, rew_seq)
            mut_states.add(ss)

        s, r, done, _ = env.step(a)
        ss = env.abst(s)

        state_seq.append(s)
        action_seq.append(a)
        rew_seq.append(r)
        steps += 1

    succ = cond(state_seq, action_seq, rew_seq)

    return mut_states, norm_states, succ, sum(rew_seq), steps

def safe_inc(counts, state, ind):
    if state not in counts:
        counts[state] = [0, 0, 0, 0]
    counts[state][ind] += 1

def update_counts(counts, mut_states, norm_states, succ):
    inds = [0, 1] if succ else [2, 3]
    for state in mut_states:
        safe_inc(counts, state, inds[1])
    for state in norm_states:
        safe_inc(counts, state, inds[0])

def safe_flex(dic, ind1, ind2, item):
    if ind1 not in dic:
        dic[ind1] = [[], []]
    dic[ind1][ind2].append(item)

def update_flex_counts(flex_counts, mut_states, norm_states, tot_rew):
    for state in mut_states:
        safe_flex(flex_counts, state, 1, tot_rew)
    for state in norm_states:
        safe_flex(flex_counts, state, 0, tot_rew)

def flex_to_counts(flex_counts, thresh):
    counts = {}
    for abs_state, (norm_rews, mut_rews) in flex_counts.items():
        counts[abs_state] = [
            int(sum(rew >= thresh for rew in norm_rews)),
            int(sum(rew >= thresh for rew in mut_rews)),
            int(sum(rew < thresh for rew in norm_rews)),
            int(sum(rew < thresh for rew in mut_rews))
            ]
    return counts
