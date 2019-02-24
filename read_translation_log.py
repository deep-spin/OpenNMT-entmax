#!/usr/bin/env python

import argparse
from os.path import basename, dirname
from itertools import groupby
from collections import defaultdict
import re
import math
import pandas as pd

EPSILON = '-100000002004087734272.0000'

def parse_beam(beam_lines):
    return [math.exp(float(beam.split(None, 1)[0][1:-1]))
            for beam in beam_lines]

def kind_of_line(line):
    if line.startswith('['):
        return 'beam'
    elif line.startswith('SENT'):
        return 'pred'
    else:
        return 'neither'

def parse_log(path):
    # attn, output = config(path)
    # find out how many hypotheses there are!
    # this requires some clever iteration over the files
    beams = []
    lang_index = []
    beams_by_lang = defaultdict(list)
    last_language = None
    with open(path) as f:
        for name, group in groupby(f, kind_of_line):
            if name == 'beam':
                # beams.append(parse_beam(group))
                beams_by_lang[last_language].append(parse_beam(group))
            elif name == 'pred':
                line = next(group)
                last_language = next(re.finditer(r'([a-z]|-)+', line)).group(0)
                # lang_index.append(lang)
    # return beams, lang_index
    return beams_by_lang


def totally_sparse_rate(beams):
    return sum(b[0] == 1 for b in beams) / len(beams)


def avg_beam_prob(beams):
    return sum(sum(b) for b in beams) / len(beams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    opt = parser.parse_args()
    beams = parse_log(opt.log)
    print(beams.keys())
    print(len(beams))
    data = defaultdict(list)
    for k, v in beams.items():
        data['language'].append(k)
        data['samples'].append(len(v))
        data['single hypothesis'].append(totally_sparse_rate(v))
        data['avg beam probability'].append(avg_beam_prob(v))
    df = pd.DataFrame(data)
    print(df)
    print(df.mean())
    print(df.iloc[df['single hypothesis'].idxmin()])
    print(df.loc[df['language'] == 'english'])
    #print(len(beams))
    #print(len(set(lang_index)))
    #print(totally_sparse_rate(beams))


if __name__ == '__main__':
    main()
