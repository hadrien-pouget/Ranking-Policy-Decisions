
ALL_CONDS = ['score[#]']

def get_cond(name):
    if name[:5] == 'score':
        if name[5:7] == 'gt':
            score = float(name[7:])
            return score_based(score, True)
        elif name[5:] == '_auto':
            return lambda x, y, z: True
        else:
            score = float(name[5:])
            return score_based(score, False)

    print(name, "condition not found, try another from:\n", "\n".join(ALL_CONDS))
    exit()

def score_based(score, strict):
    def is_higher(states, acts, rews):
        return sum(rews) >= score

    def is_strictly_higher(states, acts, rews):
        return sum(rews) > score

    if strict:
        return is_strictly_higher
    else:
        return is_higher
