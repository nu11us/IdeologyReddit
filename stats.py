import json
import random
import sys
from os import path
from collections import defaultdict


def p(jsonfile):
    with open(jsonfile) as d:
        x = 0
        for line in d:
            x += 1
    return x

def stat(fl):
    d = defaultdict(int)
    header = True
    ln = 0
    date = None
    with open(fl) as f:
        for line in f:
            ln += 1
            if header:
                header = False
            else:
                if line.count(',') > 5:
                    d[line.split(',')[1:2][0].lower()] += 1
                    date = line.split(',')[2:3][0]
                else:
                    print(ln, date)
    for elem in d:
        print(elem, "{}%".format(round(100*d[elem]/ln,2)), d[elem])
    print(ln)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_name = str(sys.argv[1])
        stat(file_name)
