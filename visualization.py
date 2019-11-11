#! /usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def save_img(fname):
    i = 2
    g = nx.MultiDiGraph()

    with open(fname + '.txt', 'r') as f:
        lines = f.readlines()

        while i < len(lines):
            if lines[i][0] == '#':
                s, t, m, n = tuple(map(int, lines[i][2:].split(' ')))
                txt = [l.split() for l in lines[i+1 : i+1+m]]
                connection = np.array(txt, dtype=np.float64).reshape((m, n))

                for p in range(m):
                    for q in range(n):
                        g.add_edge((s, q), (t, p), color=connection[p, q])

                i += m + 1
            else:
                raise ValueError

    # Extract a list of edge colors
    edges = g.edges()
    edge_colors = [g[u][v][0]['color'] for u, v in edges]

    # define fixed positions of nodes
    nodes = g.nodes()
    fixed_pos = {}
    for n in nodes:
        # x, y = tuple(map(int, n.split(',')))
        fixed_pos.update({n: n})

    # get the nx positions
    pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

    # draw
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    nx.draw(g, pos=pos, edge_cmap=plt.get_cmap('Greys'), edge_color=edge_colors,
            connectionstyle='arc3,rad=0.1', ax=ax)
    fig.savefig(fname, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    fname = 'results/%d.txt'
    for i in range(sys.argv[1]):
        pass # TODO: Visualization