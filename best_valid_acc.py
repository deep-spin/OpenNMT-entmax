#!/usr/bin/env python

import sys

print(max(float(line.strip().split()[-1]) for line in sys.stdin if 'Validation accuracy' in line))
