#!/usr/bin/env python

import sys
import argparse
from itertools import cycle
import pandas as pd
from os.path import basename, split
import re


func_names = {'so': 'softmax', 'sp': 'sparsemax', 'ts': '1.5-entmax'}


def language(path):
    return basename(path).split('.')[0]


def attn(path):
    return func_names[path.split('/')[-3].split('-')[0]]


def out(path):
    return func_names[path.split('/')[-3].split('-')[1]]


def run(path):
    return path.split('/')[-2]


def morph_num(line):
    # for accuracy and levenshtein distance
    return float(line.strip().split()[-1])


def postprocess_morph(data_dict):
    table = pd.DataFrame(data_dict)
    table['Acc.'] = table['acc'].astype(float)
    table['Lev. Dist.'] = table['lev'].astype(float)
    table['language'] = table['path'].apply(language)
    table['Attention'] = table['path'].apply(attn)
    table['Output'] = table['path'].apply(out)
    table['run'] = table['path'].apply(run)
    
    return table


def results_table(path, task, columns):
    data = {col: [] for col in columns}
    with open(path) as f:
        for col, line in zip(cycle(columns), f):
            data[col].append(line.strip())
    df = pd.DataFrame(data)
    df['Attention'] = df['path'].apply(attn)
    df['Output'] = df['path'].apply(out)
    df['run'] = df['path'].apply(run)

    if task == 'morph':
        df['Acc.'] = df['acc'].apply(morph_num)
        #df['Lev. Dist.'] = df['lev'].apply(morph_num)
        df['language'] = df['path'].apply(language)
    return df


def setting_results(df):
    single_runs = df.loc[df['run'] != 'ensemble']
    singles_avg = single_runs.groupby(['Attention', 'Output'], sort=False).mean()
    ensemble_run = df.loc[df['run'] == 'ensemble']
    ensemble_avg = ensemble_run.groupby(['Attention', 'Output'], sort=False).mean()
    return pd.concat([singles_avg, ensemble_avg], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['mt', 'morph'])
    parser.add_argument('-results_paths', nargs='+')
    parser.add_argument('-columns', nargs='+')
    opt = parser.parse_args()
    
    results = [results_table(path, opt.task, opt.columns)
               for path in opt.results_paths]

    all_results = [setting_results(setting) for setting in results]
    print(pd.concat(all_results, axis=1).to_latex(float_format='%.2f'))


if __name__ == '__main__':
    main()
