#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy import interpolate
from math import sqrt
import matplotlib
import tempfile
import os
import subprocess 
import tikzplotlib
SPINE_COLOR = 'gray'

base_path = os.path.dirname(__file__)
output_path = os.path.join(base_path, 'resources')

def load_metrics(file):
    import csv
    metrics = defaultdict(lambda: ([], []))
    for line in csv.reader(file):
        name = line[0]
        count = int(line[1])
        times = list(map(int, line[2:(2 + count)]))
        values = list(map(float, line[(2 + count):]))
        metrics[name][0].extend(times)
        metrics[name][1].extend(values)
    return metrics


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'axes.formatter.limits': [-2, 2],
              'text.usetex': True,
              'text.latex.unicode': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

def plot_paac_training():
    with open(os.path.join(base_path, 'resources/paac-training'), 'r') as f:
        paac_data = load_metrics(f)

    fig = plt.figure()
    plt.plot(*paac_data['reward'], c='r', linewidth=0.7, label='reward')
    plt.xlabel('frames')
    plt.ylabel('return')
    plt.grid(linestyle = '--')
    #plt.title('PAAC training cummulative reward')
    #plt.xlim(-, 1.2e7)
    ax = plt.gca()
    #plt.xlim(-, 1.2e7)
    plt.tight_layout()
    format_axes(ax)
    plt.savefig(os.path.join(output_path,"paac-reward.pdf"))
    plt.savefig(os.path.join(output_path,"paac-reward.eps"))

    tikzplotlib.clean_figure()
    tikzplotlib.save(os.path.join(output_path, "paac-reward.tex"))

def crop_data(*args):
    def crop_metric(*args):
        maxstep = min(max(x[0]) for x in args)
        return [([x[0] for x in zip(*a) if x[0] <= maxstep], [x[1] for x in zip(*a) if x[0] <= maxstep]) for a in args]
    b = tuple([dict() for _ in args])
    for k in args[0].keys():
        for a, nrow in zip(b,crop_metric(*[x[k] for x in args])):
            a[k] = nrow
    return b 

def plot_neat_training():
    with open(os.path.join(base_path, 'resources/neat-results.csv'), 'r') as f:
        neat_data = load_metrics(f)
    with open(os.path.join(base_path, 'resources/neat-recurrent-results.csv'), 'r') as f:
        neat_rec_data = load_metrics(f)
    neat_data, neat_rec_data = crop_data(neat_data, neat_rec_data) 
    
    fig = plt.figure()
    plt.plot(*neat_data['reward'], c='b', linewidth=1.0, label='feed-forward')
    plt.plot(*neat_rec_data['reward'], c='r', linewidth=1.0, label='recurrent')
    plt.xlabel('generation')
    plt.ylabel('return')
    plt.legend()
    plt.grid(linestyle = '--')
    #plt.title('PAAC training cummulative reward')
    #plt.xlim(-, 1.2e7)
    ax = plt.gca()
    #plt.xlim(-, 1.2e7)
    plt.tight_layout()
    format_axes(ax)
    plt.savefig(os.path.join(output_path,"neat-reward.pdf"))
    plt.savefig(os.path.join(output_path,"neat-reward.eps"))

    tikzplotlib.clean_figure()
    tikzplotlib.save(os.path.join(output_path, "neat-reward.tex"))

def output_latex_network(name):
    import utils.plotnet as p
    import shutil

    path = os.path.join(base_path, 'networks')
    subprocess.check_call(['pdflatex', name], cwd=path)
    subprocess.check_call(['pdftops','-eps','%s.pdf' % name,'%s.eps' % name], cwd=path)
    #subprocess.check_call(['pdflatex', '-output-format', 'dvi', 'network'], cwd=tmpd)
    #subprocess.check_call(['dvips','-E','network.dvi','-o','network.eps'], cwd=tmpd)
    shutil.copy(os.path.join(path, '{name}.eps'.format(name = name)), os.path.join(output_path, '{name}.eps'.format(name = name)))
    shutil.copy(os.path.join(path, '%s.pdf' % name), os.path.join(output_path, '%s.pdf' % name))

def output_latex_networks():
    output_latex_network('modules')
    output_latex_network('conv-base')
    output_latex_network('aux-head')
    output_latex_network('pc-head')

def clip_plot_data(data1, data2):
    t1, d1 = data1
    t2, d2 = data2
    tmax = min(max(t1), max(t2))

    limit1 = sum(np.array(t1) <= tmax)
    t1, d1 = t1[:limit1], d1[:limit1]

    limit2 = sum(np.array(t2) <= tmax)
    t2, d2 = t2[:limit2], d2[:limit2]
    return (t1, d1), (t2, d2)

if __name__ == '__main__':
    columns = 1
    latexify(fig_width=4.26791486111)
    plot_paac_training()
    plot_neat_training()


    # output_latex_networks()
