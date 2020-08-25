import os

import numpy as np
import matplotlib.pyplot as plt

from scoring import ST_COLOURS

def draw_interpol_results(logger, score_types, which_x, which_ys,
    trans_x=lambda x: x, x_name=None, y_names=None,
    hlines=False, x_fracs=False, y_fracs=False, y_offset=0, saveloc=None,
    smooth=False, combine_sbfl=False):
    ''' Given score types, which x axis to use, and several y axes: makes as many graphs as
    y axes. Plots results for all score_types on one graph.

    trans_x: is function applied to all x-axis values
    x_name: replaces x-axis name if provided
    hlines: is created from data at end of y lists
    y_fracs: makes y-axis fraction of baseline values (end of y lists) '''

    data = logger.data['interpol'][0]
    xs = [list(map(trans_x, data[st][which_x]))[:-1] for st in score_types]
    yss = [[data[st][wy] for st in score_types] for wy in which_ys]

    if x_fracs:
        xs = [[val/max(x) for val in x] for x in xs]
    if y_offset:
        new_yss = []
        for ys in yss:
            new_yss.append([[val-y_offset for val in vals] for vals in ys])
        yss = new_yss

    # Get final values for baseline and remove them
    hvals = [np.mean([vals[-1] for vals in ys]) for ys in yss]
    yss = [[vals[:-1] for vals in ys] for ys in yss]

    if y_fracs:
        new_yss = []
        for ys, hval in zip(yss, hvals):
            new_yss.append([[(val/hval) for val in vals] for vals in ys])
        yss = new_yss
        hvals = [1] * len(which_ys)

    if not hlines:
        hvals = [None] * len(which_ys)

    x_name = x_name if x_name is not None else logger.interpol_cols[which_x]
    y_names = y_names if y_names is not None else[logger.interpol_cols[wy] for wy in which_ys]

    saveloc = saveloc if saveloc is not None else logger.fileloc
    savelocs = [os.path.join(saveloc, x_name + '_' + y_name + '_' + '_'.join(score_types)) for y_name in y_names]

    for ys, y_name, hv, sl in zip(yss, y_names, hvals, savelocs):
        draw_curves(xs, ys, x_name, y_name, score_types, hval=hv, sloc=sl, smooth=smooth, combine_sbfl=combine_sbfl)

def smoothing(x, y, spline=False):
    # Make sure x and y are sorted properly first!

    if spline: # Currently fails if 2 x-values are the same
        from scipy import interpolate
        smoothness = 0 # Amount of smoothness - 0 is usually enough
        x_smooth = np.linspace(x.min(), x.max(), 300)
        tck = interpolate.splrep(x, y, s=smoothness)
        y_smooth = interpolate.splev(x_smooth, tck, der=0)
        return x_smooth, y_smooth

    window = 3 # Must be odd
    new_y = []
    for i in range(len(y)):
        w = min(int(window/2), i, len(y)-i-1) # How far can the window extend to the right or left
        new_y.append(sum(y[i-w:i+w+1])/((2*w)+1))

    return x, new_y

def draw_curves(xs, ys, x_name, y_name, score_types, hval=None, sloc=None, smooth=False, combine_sbfl=False):
    plt.clf()

    ## Combines SBFL lines while only taking point which improve on the previous
    if combine_sbfl:
        new_xs, new_ys, new_st, sbfl_x, sbfl_y = [], [], [], [], []
        for x, y, st in zip(xs, ys, score_types):    
            if st in ["zoltar", "tarantula", "wongII", "ochiai"]:
                sbfl_x.append(x)
                sbfl_y.append(y)
            else:
                new_xs.append(x)
                new_ys.append(y)
                new_st.append(st)

        sbfl_x, sbfl_y = combine_lines(sbfl_x, sbfl_y, only_improve=True)
        new_xs.append(sbfl_x)
        new_ys.append(sbfl_y)
        new_st.append('SBFL')

        xs, ys, score_types = new_xs, new_ys, new_st
    ##

    xmax = 0
    for x, y, st in zip(xs, ys, score_types):
        xmax = max(xmax, max(x))
        srt = list(zip(x, y))
        srt.sort(key=lambda x: x[0])
        x, y = [e[0] for e in srt], [e[1] for e in srt]

        if smooth:
            x, y = smoothing(x, y)

        ## Turn into percentages
        if '%' in y_name:
            y = [val*100 for val in y]
        if '%' in x_name:
            x = [val*100 for val in x]
        ##

        ## Makes sure non-SBFL lines also only taking point that improve
        if st in ['rand', 'freqVis', 'tarantula', 'ochiai', 'zoltar', 'wongII']:
            x, y = only_improve(x, y)
        ##

        if st == 'rand':
            plt.plot(x, y, linestyle='dashed', label=st, color=ST_COLOURS[st], linewidth=3)
        else:
            plt.plot(x, y, label=st, color=ST_COLOURS[st], linewidth=3)

    if hval is not None:
        # plt.plot([0, xmax], [hval, hval], linestyle='dashed', label='baseline')
        plt.axhline(y=hval, xmin=0, xmax=1, linestyle='dotted', label='baseline', color='red')

    # axes = plt.axes()
    # axes.set_ylim([0, 110])

    plt.legend()
    plt.ylabel(y_name, size=17)
    plt.xlabel(x_name, size=17)
    plt.tight_layout()

    if sloc is not None:
        plt.draw()
        plt.savefig(sloc+'.eps', format='eps')
        plt.savefig(sloc+'.png', format='png')
    else:
        plt.show()

    plt.clf()

def combine_lines(xs, ys, only_improve=False):
    """ Take list of lists for xs and ys (for each score type)
    and use best in each point. """

    all_points = [(x_val, y_val) for x, y in zip(xs, ys) for x_val, y_val in zip(x, y)]
    all_points.sort(key=lambda tup: tup[0])

    xmax = all_points[-1][0]

    combined_x = [all_points[0][0]]
    combined_y = [all_points[0][1]]

    if only_improve:
        for x_val, y_val in all_points[1:]:
            if x_val == combined_x[-1]:
                combined_y[-1] = max(y_val, combined_y[-1])
            elif y_val > combined_y[-1]:
                combined_x.append(x_val)
                combined_y.append(y_val)
        combined_x.append(xmax)
        combined_y.append(combined_y[-1])
    else:
        for x_val, y_val in all_points[1:]:
            if x_val == combined_x[-1]:
                combined_y[-1] = max(y_val, combined_y[-1])
            else:
                combined_x.append(x_val)
                combined_y.append(y_val)

    return combined_x, combined_y

# Redundant with above
def only_improve(xs, ys):
    """ Takes list of x values and list of y values """
    new_xs, new_ys = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if x == new_xs[-1]:
            new_ys[-1] = max(new_ys[-1], y)
        elif y > new_ys[-1]:
            new_xs.append(x)
            new_ys.append(y)
    if new_xs[-1] < xs[-1]:
        new_xs.append(xs[-1])
        new_ys.append(new_ys[-1])

    return new_xs, new_ys

def cartpole_graphs(fileloc, scores, score_types):
    import matplotlib.pyplot as plt 
    import numpy as np
    from scoring import ST_COLOURS

    x_names = ['CartPos', 'CartSpeed', 'PoleAng', 'PoleSpeed']
    for i in range(4):
        plt.clf()
        for st in score_types:
            scrs = scores[st]
            vals = {}
            for s, sc in scrs:
                s = s[1:-1].split(',')
                x = float(s[i])
                if x in vals:
                    vals[x].append(sc)
                else:
                    vals[x] = [sc]

            for s, sc in vals.items():
                vals[s] = np.mean(sc)

            vals = list(vals.items())
            vals.sort(key=lambda tup:tup[0])
            vals = list(zip(*vals))

            diff = max(vals[1])-min(vals[1])
            if diff > 0:
                vals[1] = [(i-min(vals[1]))/diff for i in vals[1]]
            else:
                vals[1] = [0 for i in vals[1]]
            plt.plot(vals[0], vals[1], label=st, color=ST_COLOURS[st], linewidth=3)

        plt.legend()
        plt.ylabel('Score', size=17)
        plt.xlabel(x_names[i], size=17)
        plt.tight_layout()
        plt.draw()
        plt.savefig(fileloc + str(i) + '.png')

        # plt.clf()
        # for st in score_types:
        #     scrs = scores[st]
        #     xs, ys = [], []
        #     for s, sc in scrs:
        #         s = s[1:-1].split(',')
        #         x = float(s[i])
        #         xs.append(x)
        #         ys.append(sc)

        #     diff = max(ys)-min(ys)
        #     if diff > 0:
        #         ys = [(i-min(ys))/diff for i in ys]
        #     else:
        #         ys = [0 for i in ys]
        #     plt.scatter(xs, ys, s=2, label=st, color=ST_COLOURS[st])

        # plt.legend()
        # plt.ylabel('Score', size=17)
        # plt.xlabel(x_names[i], size=17)
        # plt.tight_layout()
        # plt.draw()
        # plt.savefig(fileloc + str(i) + '_scatter.png')
