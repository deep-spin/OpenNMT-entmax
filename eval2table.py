#!/usr/bin/env python

import sys
import argparse
from itertools import cycle
import pandas as pd
from os.path import basename, split
import re


func_names = {'so': '1', 'sp': '2', 'ts': '1.5'}


def language(path):
    return basename(path).split('.')[0]


#def attn(path):
#    return func_names[path.split('/')[-3].split('-')[0]]


def attn(path):
    return func_names[next(re.finditer(r'(so|sp|ts)-(so|sp|ts)', path)).group(0).split('-')[0]]


#def out(path):
#    return func_names[path.split('/')[-3].split('-')[1]]

def out(path):
    return func_names[next(re.finditer(r'(so|sp|ts)-(so|sp|ts)', path)).group(0).split('-')[1]]


def morph_run(path):
    return path.split('/')[-2]


def mt_run(path):
    return path.split('/')[-1].split('-')[0]


def bleu(line):
    return float(next(re.finditer(r'[0-9]+\.[0-9]+', line)).group(0))


def morph_num(line):
    # for accuracy and levenshtein distance
    return float(line.strip().split()[-1])


def results_table(path, task, columns):
    data = {col: [] for col in columns}
    with open(path) as f:
        for col, line in zip(cycle(columns), f):
            data[col].append(line.strip())
    df = pd.DataFrame(data)
    df['Output'] = df['path'].apply(out)
    df['Attention'] = df['path'].apply(attn)

    if task == 'morph':
        df['run'] = df['path'].apply(morph_run)
        df['Acc.'] = df['acc'].apply(morph_num)
        #df['Lev. Dist.'] = df['lev'].apply(morph_num)
        df['language'] = df['path'].apply(language)
    else:
        df['run'] = df['path'].apply(mt_run)
        df['BLEU'] = df['bleu'].apply(bleu)
    return df


def setting_results(df):
    single_runs = df.loc[df['run'] != 'ensemble']
    singles_avg = single_runs.groupby(['Output', 'Attention']).mean()
    ensemble_run = df.loc[df['run'] == 'ensemble']
    ensemble_avg = ensemble_run.groupby(['Output', 'Attention']).mean()
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
