#!/usr/bin/env python

import sys
import re

for line in sys.stdin:
    sys.stdout.write(re.sub(r"@@ ", "", line))
