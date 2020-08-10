import json
import random

def clean(inp, out):
    o = open(out, 'w+')
    with open(inp) as f:
        header = True
        for line in f:
            if header:
                o.write(line)
                header = False
            else:
                subreddit = line.split(',')[1:2][0]
                score = line.split(',')[6:7][0] 
                if int(score) > 0 and subreddit != "politics":
                    o.write(line)

def shrink(inp, out, percent):
    o = open(out, 'w+')
    with open(inp) as f:
        header = True
        for line in f:
            if header:
                o.write(line)
                header = False
            else:
                if random.random() < percent:
                    o.write(line)

shrink("politics.csv", "politics25.csv", 0.25)