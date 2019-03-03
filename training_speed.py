#!/usr/bin/env python

import sys

src_tok, tgt_tok = [], []

for line in sys.stdin:
    line = line.strip()
    if line.endswith('sec'):
        s, t = line.split()[-4].split('/')
        src_tok.append(float(s))
        tgt_tok.append(float(t))

print('src wps: ', sum(src_tok) / len(src_tok))
print('tgt wps: ', sum(tgt_tok) / len(tgt_tok))
        
