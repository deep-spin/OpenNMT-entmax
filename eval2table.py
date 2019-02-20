#!/usr/bin/env python

import sys
import argparse
from itertools import cycle
import pandas as pd
from os.path import basename, split
import re


def language(path):
    return basename(path).split('.')[0]


def attn(path):
    return path.split('/')[-3].split('-')[0]


def out(path):
    return path.split('/')[-3].split('-')[1]


def run(path):
    return path.split('/')[-2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('columns', nargs='+')
    opt = parser.parse_args()
    data = {col: [] for col in opt.columns}
    for col, line in zip(cycle(opt.columns), sys.stdin):
        data[col].append(line.strip().split('\t')[-1])
    table = pd.DataFrame(data)
    table['acc'] = table['acc'].astype(float)
    table['lev'] = table['lev'].astype(float)
    table['language'] = table['path'].apply(language)
    table['attn'] = table['path'].apply(attn)
    table['out'] = table['path'].apply(out)
    table['run'] = table['path'].apply(run)
    print(table.groupby(('attn', 'out')).mean())


if __name__ == '__main__':
    main()
