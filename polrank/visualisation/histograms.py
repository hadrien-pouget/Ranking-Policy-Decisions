import os

import matplotlib.pyplot as plt

def score_histogram(logger, score_types, nbins=10, saveloc=None):
    saveloc = saveloc if saveloc is not None else logger.fileloc
    for st in score_types:
        plt.clf()
        sloc = os.path.join(saveloc, 'hist_' +st)

        scores = logger.data['scores'][0][st]
        scores = [s[1] for s in scores]
        plt.hist(scores, bins=nbins)

        plt.ylabel("Number of States", size=17)
        plt.xlabel("Score", size=17)
        plt.tight_layout()

        plt.draw()
        plt.savefig(sloc+'.eps', format='eps')
        plt.savefig(sloc+'.png', format='png')
