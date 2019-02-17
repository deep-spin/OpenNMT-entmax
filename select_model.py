#!/usr/bin/env python

import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-metric', default='accuracy')
    opt = parser.parse_args()
    scores = []
    paths = []
    f = max if opt.metric == 'accuracy' else min
    m = 'Validation ' + opt.metric
    for line in sys.stdin:
        if m in line:
            scores.append(float(line.strip().split()[-1]))
        elif 'Saving checkpoint' in line:
            paths.append(line.strip().split()[-1])

    sys.stdout.write(f(zip(scores, paths))[1] + '\n')

if __name__ == '__main__':
    main()
        
