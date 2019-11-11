#! /usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_connections(fname):
    data = {}
    i = 1
    with open(fname, 'r') as f:
        lines = f.readlines()
        data.update({'score': float(lines[0])})

        while i < len(lines):
            if lines[i][0] == '#':
                s, t, m, n = tuple(map(int, lines[i][2:].split(' ')))
                txt = [l.split() for l in lines[i+1 : i+1+m]]
                data.update({'%d->%d' %(s , t): np.array(txt, dtype=np.float64).reshape((m, n))})
                i += m + 1
            else:
                raise ValueError
    return data

if __name__ == '__main__':
    fname = 'results/%d.txt'
    for i in range(sys.argv[1]):
        data = load_connections(fname % i)
        # TODO: Visualization