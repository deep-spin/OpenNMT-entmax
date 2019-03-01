#!/usr/bin/env python

import argparse
import glob
from os.path import basename
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches


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
                draw_square(ax, jj, ii, color="#aaaaaa", lw=2, alpha=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-plot', default='seachange.txt')
    parser.add_argument('-outpath', default='attn.png')
    parser.add_argument('-fontsize', type=int, default=11)
    opt = parser.parse_args()

    attns = load(opt.plot)
    src = "Aber wir beginnen , eine Veränderung zu sehen .".split()
    pred = "But we start to see a change . </s>".split()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.cm.OrRd
    cmap.set_bad('white')
    #attns = np.ma.masked_where(attns == 0, attns)
    cax = ax.matshow(attns, cmap=cmap)
    draw_all_squares(ax, attns)
    # fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + list(src), rotation=55, fontsize=opt.fontsize)
    ax.set_yticklabels([''] + list(pred), fontsize=opt.fontsize)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(opt.outpath, bbox_inches='tight')


if __name__ == '__main__':
    main()
