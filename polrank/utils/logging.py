import os
import shutil
import json
import csv
import pickle as pkl

from interpolating import INTERPOL_COLS
import elements.envs as envs
import elements.policies as policies
import elements.conditions as conditions

RESULTS_LOC = 'results'

def load_sev_loggers(prefix, skip_load=True):
    """ Load all loggers starting with 'prefix_' 
    Skips loading actual object from config by default, so these can't 
    be used for more experiments """
    fls = [fl for fl in os.listdir('results') if fl.startswith('{}_'.format(prefix))]
    loggers = [Logger(fl) for fl in fls]
    for logger in loggers:
        logger.load_config({'skip_load': skip_load})
        logger.load_results()
    return loggers

class Logger():
    """ Keeps track of all important data, including hyperparameters and results.
    Provides methods for saving and loading these, as well as updating them. """
    def __init__(self, fileloc, load_loc=None):
        self.fileloc = os.path.join(RESULTS_LOC, fileloc)

        if load_loc is not None and os.path.isdir(self.fileloc):
            print("Trying to load config from a previous run, but {} already exists.".format(self.fileloc))
            exit()
        else:
            os.makedirs(self.fileloc, exist_ok=True)

        # Copy over config from load location, if provided
        if load_loc is not None:
            load_loc = os.path.join(RESULTS_LOC, load_loc)
            shutil.copyfile(os.path.join(load_loc, 'config.json'), os.path.join(self.fileloc, 'config.json'))
            print("Config successfully copied from {}".format(load_loc))

        self.data = {
            'counts': [None, 'json'], # {state: [ep, np, ef, nf]}
            'scores': [None, 'json'], # {score_type: [(state, score)]} in descending order by score
            'interpol': [None, 'csv'], # {score_type: inds, avgs, vars, chks, mut_props, mut_ns}
            'logs': [None, 'json'] # For storing any info we want
        }

        self.filetools = {
            'csv': (self.load_csv, self.save_csv),
            'json': (self.load_json, self.save_json),
            'obj': (self.load_obj, self.save_obj)
        }

        self.interpol_cols = INTERPOL_COLS

        self.config = {}
        self.save_vars = [
            "env_name",
            "env_seed",
            "pol_name",
            "pol_d_name",
            "cond_name",
            "n_runs",
            "mut_prob",
            "score_types",
            "n_inc",
            "n_test",
            "abst_type",
            "max_steps",
            "no_det"]

    ### Config ###
    # Config will contain all parameters in args, but only save those in save_vars across sessions.
    # When loading config, priority is given to loaded values, and blanks are filled with args.
    def filter_vars(self, dic):
        """ Select variables that we want to save in config """
        return {k: v for k, v in dic.items() if k in self.save_vars}

    def load_from_names(self):
        """ Load env, policies and conditions based on the provided names """
        device = self.config['device']
        self.config['env'] = envs.get_env(
            self.config['env_name'],
            self.config['use_seed'],
            device=device,
            abst_type=self.config['abst_type'],
            vis_type=self.config['vis_type'],
            max_steps=self.config['max_steps'])
        self.config['pol'] = policies.get_pol(self.config['pol_name'], self.config['env'], device)
        self.config['pol_d'] = policies.get_pol(self.config['pol_d_name'], self.config['env'], device)
        self.config['cond'] = conditions.get_cond(self.config['cond_name'])

    def update_config(self, dic, skip_load=False):
        self.config.update(dic)
        if not skip_load:
            self.load_from_names()

    def init_config(self, args):
        """ If a config already exists, load it, and fill in any blanks with args.
        Otherwise, use the args supplied. """
        if not self.load_config(vars(args)):
            self.update_config(vars(args))
            self.dump_config()

    def dump_config(self):
        """ Save config. Only self.save_vars are saved """
        c = json.dumps(self.filter_vars(self.config), sort_keys=True, indent=4, separators=(',', ':'))
        configloc = os.path.join(self.fileloc, 'config.json')
        with open(configloc, 'w') as f:
            f.write(c)

    def load_config(self, dic):
        """ Return whether successful or not """
        configloc = os.path.join(self.fileloc, 'config.json')
        try:
            with open(configloc, 'r') as f:
                config = json.load(f)
            dic.update(config)
            self.update_config(dic, skip_load=dic['skip_load'])
            print("config successfully loaded from", self.fileloc)
            return True
        except FileNotFoundError:
            print("config not found at {}, making new config".format(self.fileloc))
            return False

    ### Data helpers ###
    def cols_to_rows(self, cols):
        ''' For saving CSVs, change list of columns to list of rows.
        Also works for going back (rows to cols) '''
        nrows = range(len(cols[0]))
        rows = [[c[i] for c in cols] for i in nrows]
        return rows

    def load_csv(self, lloc):
        with open(lloc, 'r') as f:
            lines = []
            reader = csv.reader(f)
            for line in reader:
                l = list(map(lambda x: float(x), line))
                lines.append(l)
            return lines

    def load_json(self, lloc):
        with open(lloc, 'r') as f:
            return json.load(f)

    def load_obj(self, lloc):
        with open(lloc, 'rb') as f:
            return pkl.load(f)

    def save_csv(self, sloc, lines, mode='w'):
        with open(sloc, mode, newline='') as f:
            writer = csv.writer(f)
            for line in lines:
                writer.writerow(line)

    def save_json(self, sloc, dic):
        js = json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))
        with open(sloc, 'w') as f:
            f.write(js)

    def save_obj(self, sloc, obj):
        with open(sloc, 'wb') as f:
            pkl.dump(f, obj)

    def is_done(self, name):
        """ Returns if type of data 'name' (in self.data) has already been collected """
        return self.data[name][0] is not None

    ### Load data ###
    def load_results(self):
        for name, (_, ftype) in self.data.items():
            loader = self.filetools[ftype][0]
            if name == 'interpol':
                score_types = [f[9:][:-4] for f in os.listdir(self.fileloc) if ('interpol_' in f) and ('.csv' in f)]
                llocs = [os.path.join(self.fileloc, name + '_' + st + '.' + ftype) for st in score_types]

                if len(llocs) == 0:
                    print("No {} loaded from {}".format(name, self.fileloc))
                    continue

                self.data[name][0] = {}
                for st, lloc in zip(score_types, llocs):
                    rows = loader(lloc)
                    cols = self.cols_to_rows(rows)
                    self.data[name][0][st] = cols
                    print("{} loaded from {}".format(name, self.fileloc))

            else:
                lloc = os.path.join(self.fileloc, name + '.' + ftype)
                try:
                    res = loader(lloc)
                    self.data[name][0] = res
                    print("{} loaded from {}".format(name, self.fileloc))
                except FileNotFoundError:
                    print("No {} loaded from {}".format(name, self.fileloc))

    ### Save data ###
    def dump_results(self):
        for name, (d, ftype) in self.data.items():
            saver = self.filetools[ftype][1]
            if d is not None:
                if name == 'interpol':
                    for st, cols in d.items():
                        sloc = os.path.join(self.fileloc, name + '_' + st + '.' + ftype)
                        rows = self.cols_to_rows(cols)
                        saver(sloc, rows)
                else:
                    sloc = os.path.join(self.fileloc, name + '.' + ftype)
                    saver(sloc, d)

    ### Update data ###
    def update_counts(self, ncounts, addn=0):
        counts = self.data['counts'][0] if self.data['counts'][0] is not None else {}
        for st, ncs in ncounts.items():
            try:
                cs = counts[st]
                cs = [cs[i] + ncs[i] for i in range(4)]
                counts[st] = cs
            except KeyError:
                counts[st] = ncs

        self.data['counts'][0] = counts
        self.config['n_runs'] += addn

    def update_scores(self, scores):
        """ Update tracked scores. scores should be a
        dict of scores """
        if self.data['scores'][0] is None:
            self.data['scores'][0] = scores
        else:
            self.data['scores'][0].update(scores)

        for st in scores:
            if st not in self.config['score_types']:
                self.config['score_types'].append(st)

    def update_interpolation(self, interpol):
        """ Update intepolation results """
        if self.data['interpol'][0] is None:
            self.data['interpol'][0] = interpol
        else:
            self.data['interpol'][0].update(interpol)

    def update_logs(self, nlog):
        """ Update log """
        if self.data['logs'][0] is None:
            self.data['logs'][0] = nlog
        else:
            self.data['logs'][0].update(nlog)

    def update_cond(self, new_cond_name):
        self.config.update({'cond_name': new_cond_name})
        self.config['cond'] = conditions.get_cond(self.config['cond_name'])
        self.dump_config()
