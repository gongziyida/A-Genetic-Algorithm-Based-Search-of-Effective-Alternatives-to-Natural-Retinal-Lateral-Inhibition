#! /usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def save_img(fname):
    i = 2
    g = nx.MultiDiGraph()
    counter = 0
    n_cells = {}

    with open(fname + '.txt', 'r') as f:
        lines = f.readlines()

        while i < len(lines):
            if lines[i][0] == '#':
                counter += 1

                s, t, m, n = tuple(map(int, lines[i][2:].split(' ')))

                if not s in n_cells:
                    n_cells.update({s: n})
                if not t in n_cells:
                    n_cells.update({t: m})

                txt = [l.split() for l in lines[i+1 : i+1+m]]

                connection = np.array(txt, dtype=np.float64).reshape((m, n))

                for p in range(m):
                    for q in range(n):
                        if np.abs(connection[p, q]) > 0.1:
                            g.add_edge((s, q), (t, p), color=connection[p, q])

                i += m + 1
            else:
                raise ValueError

    if counter == 0:
        print('No interneuron for', fname)
        return

    # Extract a list of edge colors
    edges = g.edges()
    if len(edges) == 0:
        print('No non-trivial connections for', fname)
        return

    edge_colors = [g[u][v][0]['color'] for u, v in edges]

    # define fixed positions of nodes
    nodes = g.nodes()
    fixed_pos = {}
    for n in nodes:
        fixed_pos.update({n: (n[0], (n[1] - n_cells[n[0]]/2) * 2)})

    # get the nx positions
    pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

    # draw
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    nx.draw(g, pos=pos, node_size=100,
            edge_cmap=plt.get_cmap('bwr'), edge_color=edge_colors, edge_vmin=-1, edge_vmax=1,
            connectionstyle='arc3,rad=0.1', ax=ax)
    fig.savefig(fname, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    log = np.loadtxt('results/LOG', dtype=np.float64, delimiter=' ', skiprows=1)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    for i in range(1, 4):
        ax.plot(log[:, 0], log[:, i])
    ax.legend(['min', 'max', 'avg'])
    fig.savefig('results/performance.png', bbox_inches='tight')

    fname = 'results/%d'
    for i in range(int(sys.argv[1])):
        save_img(fname % i)
