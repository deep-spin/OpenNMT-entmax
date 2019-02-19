#!/usr/bin/env python

import argparse
from itertools import groupby
import re
from os.path import join


def split_predictions(test_src, test_pred):
    with open(test_src) as f:
        langs = [line.split(None, 1)[0] for line in f]
    with open(test_pred) as f:
        preds = {lang: [g[1].strip() for g in group] 
                 for lang, group in groupby(zip(langs, f), key=lambda x: x[0])}
    return preds


def make_out(sigmorphon_path, lang_pred, out_path):
    with open(sigmorphon_path) as sig_f, open(out_path, 'w') as out_f:
        for line, pred in zip(sig_f, lang_pred):
            lemma, _, tags = line.strip().split('\t')
            inflection = re.sub(r'\s', '', pred.strip())
            inflection = re.sub(r'<space>', ' ', inflection)
            out_f.write('\t'.join([lemma, inflection, tags]) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')  # all-language test.src file
    parser.add_argument('tgt')  # all-language predictions file
    parser.add_argument('out_dir')
    parser.add_argument('-sigmorphon_dir', default='/mnt/data/bpop/inflection/conll2018/task1/everything')  # directory containing sigmorphon test files.
    opt = parser.parse_args()

    # group predictions by language
    # keys are language names, values are lists of predictions
    preds = split_predictions(opt.src, opt.tgt)
    print(preds.keys())
    for lang, pred in preds.items():
        out_path = join(opt.out_dir, lang + '.pred.out')
        sig_path = join(opt.sigmorphon_dir, lang + '-test')
        make_out(sig_path, pred, out_path)



if __name__ == '__main__':
    main()
