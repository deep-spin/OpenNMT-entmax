#!/usr/bin/env python

import argparse
import glob
from os.path import basename
import torch
import numpy as np
import matplotlib

# matplotlib.use('pgf')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.colors as colors


matplotlib.rcParams['figure.constrained_layout.use'] = True

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage[T1]{fontenc}',
    r'\usepackage{times}',
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
]

# matplotlib.rcParams['font.family'] = 'serif'

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def config(path):
    return tuple(basename(path).split('-')[:2])


def load(path):
    with open(path) as f:
        return np.array([line.strip().split() for line in f], dtype=float)


def draw_square(ax, i, j, **kwargs):
    ax.add_patch(patches.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, **kwargs))


def draw_all_squares(ax, M):
    for ii in range(M.shape[0]):
        for jj in range(M.shape[1]):
            if M[ii, jj] > 0:
                draw_square(ax, jj, ii, color="#aaaaaa", lw=1, alpha=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-plot', default='seachange.txt')
    parser.add_argument('-fontsize', type=int, default=10)
    opt = parser.parse_args()

    attns = load(opt.plot)
    src = "Aber wir beginnen , eine Veränderung zu sehen .".split()
    pred = "But we start to see a change . </s>".split()

    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_subplot(111)
    cmap = plt.cm.PuOr_r  # OrRd
    cax = ax.matshow(attns, cmap=cmap,
            clim=(-1, 1),
            norm=MidpointNormalize(midpoint=0,vmin=1, vmax=1))
    draw_all_squares(ax, attns)
    # fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + list(src), rotation=45, fontsize=opt.fontsize,
            horizontalalignment='left')
    ax.set_yticklabels([''] + list(pred), fontsize=opt.fontsize)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("seachange.pdf")
    # plt.savefig(opt.outpath, bbox_inches='tight')


if __name__ == '__main__':
    main()
