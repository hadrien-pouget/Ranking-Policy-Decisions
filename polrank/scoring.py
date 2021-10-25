import random
import time

import numpy as np

from utils.timing import sec_to_str

ALL_SCORE_TYPES = ['tarantula', 'ochiai', 'zoltar', 'wongII', 'freqVis',
    'rand', 'p(f|l)', 'mymeth', 'rev_wongII']

ST_COLOURS = {st: 'C' + str(i) for i, st in enumerate(ALL_SCORE_TYPES)}
ST_COLOURS['SBFL'] = 'black'

def score(logger, score_types=None):
    config = logger.config
    score_types = score_types if score_types is not None else config['score_types']
    update_logs = logger.update_logs
    scores = score_by_type(
        logger.data['counts'][0],
        score_types,
        update_logs)
    return get_ranking(scores)

def get_ranking(scores):
    ranking = {}
    for st, scrs in scores.items():
        scrs = list(scrs.items())
        scrs.sort(key=lambda x: x[1], reverse=True)
        ranking[st] = scrs
    return ranking

def score_by_type(counts, score_types, update_logs):
    """ Score states based on counts, using score_types """
    start = time.time()

    all_scores = {}
    print("\nBeginning scoring")
    for st in score_types:
        scores = {}
        for k, v in counts.items():
            scores[k] = score_state(v, st)
        all_scores[st] = scores
        print("Done with score_type:", st)

    end = time.time()
    log = {
        'score_time': sec_to_str(end-start)
    }
    update_logs(log)

    return all_scores

def score_state(v, st):
    """ Given values and score types, calculate score """
    # v: [ep, np, ef, nf]
    if st == "tarantula":
        p1 = (v[3] / (v[1] + v[3])) if v[3] > 0 else 0 # p(lazy | fail)
        p2 = (v[2] / (v[0] + v[2])) if v[2] > 0 else 0 # p(lazy | pass)
        scr = p1 / (p1 + p2) if p1 > 0 else 0
    elif st == "ochiai":
        p1 = np.sqrt((v[3] + v[1])*(v[3] + v[2]))
        scr = v[3] / p1 if p1 > 0 else 0
    elif st == "zoltar":
        p2 = (v[3] + v[1] + v[0] + ((10000*v[3]*v[0])/v[3])) if v[3] > 0 else 0
        scr = v[3] / p2 if p2 > 0 else 0
    elif st == "wongII":
        scr = v[3] - v[2]
    elif st == "rev_wongII":
        scr = v[2] - v[3]
    elif st == "p(f|l)":
        scr = v[3] / (v[1] + v[3]) if v[3] > 0 else 0
    elif st == "mymeth":
        p1 = v[3] / (v[1] + v[3]) if v[3] > 0 else 0
        scr = (sum(v)/500) * p1
    elif st == "freqVis":
        scr = sum(v)
    elif st == "rand":
        scr = 1
    else:
        scr = 0
    return scr
