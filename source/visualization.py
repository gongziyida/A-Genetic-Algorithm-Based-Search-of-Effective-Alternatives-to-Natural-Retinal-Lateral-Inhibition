#! /usr/bin/env python3
import sys
import os
import numpy as np
# Non-interactive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

COMMON = 'red'
NO_INTERNEURON = 'black'
NO_NON_TRIVIAL_W = 'gray'

def save_img(fname):
    i = 2
    counter = 0
    n_cells = {}
    connections = {}

    with open(fname + '.tsv', 'r') as f:
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

                c = np.array(txt, dtype=np.float64).reshape((m, n))
                if c.any(): # any non-zeros
                    connections.update({(s, t): c})

                i += m + 1
            else:
                raise ValueError

    if counter == 0:
        print('No interneuron for', fname)
        return float(lines[0]), NO_INTERNEURON

    if len(connections) == 0:
        print('No non-trivial weight for', fname)
        return float(lines[0]), NO_NON_TRIVIAL_W

    fig, ax = plt.subplots(len(connections))
    fig.set_size_inches(5, 5 * len(connections))

    if len(connections) == 1:
        ax = [ax]

    for i, (s, t) in enumerate(connections.keys()):
        g = nx.MultiDiGraph()

        n = n_cells[s]
        m = n_cells[t]
        c = connections[(s, t)]

        for p in range(m):
            for q in range(n):
                if np.abs(c[p, q]) > 0.01:
                    g.add_edge((s, q), (t, p), color=c[p, q])
                else:
                    g.add_node((s, q))
                    g.add_node((t, p))

        # Extract a list of edge colors
        edges = g.edges()
        edge_colors = [g[u][v][0]['color'] for u, v in edges]

        # define fixed positions of nodes
        nodes = g.nodes()
        fixed_pos = {}
        for n in nodes:
            fixed_pos.update({n: (n[0], (n[1] - n_cells[n[0]]/2) * 30//n_cells[n[0]])})

        # get the nx positions
        pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

        # draw
        nx.draw(g, pos=pos, node_size=50, node_color='black',
                edge_cmap=plt.get_cmap('bwr'), edge_color=edge_colors, edge_vmin=-1, edge_vmax=1,
                ax=ax[i])

        ax[i].text(0.5, 1, '%d -> %d' % (s, t), transform=ax[i].transAxes, fontsize=10)

    fig.savefig(fname, dpi=200, bbox_inches='tight')

    return float(lines[0]), COMMON

def search(li, s):
    for i in li:
        if s in i:
            return i.split(' ')[1]

if __name__ == '__main__':

    path = sys.argv[1]
    max_id = int(sys.argv[2])
    max_nn = int(sys.argv[3])

    with open(os.path.join(path, 'param'), 'r') as f:
        param = f.readlines()

    iters = int(search(param, 'iter'))
    x = np.arange(1, iters + 1)

    for i in range(max_id + 1):
        # load
        log = np.loadtxt(os.path.join(path, 'log%d.tsv' % i), dtype=np.float64)

        fig, ax = plt.subplots(4, sharex='all')
        fig.set_size_inches(w=4, h=13)
        fig.tight_layout()

        for k, title in enumerate(('Total', 'Test Loss', 'AUC', 'Number of Synapses')):
            log_k = log[:, k].reshape(iters, -1)

            q = np.nanquantile(log_k, [0.25, 0.5, 0.75], axis=1)
            log_k = np.ma.masked_invalid(log_k)

            ax[k].set_title(title)

            # ax.plot(x, log.min(axis=1), color='gray', linestyle='dotted', label='min')
            ax[k].plot(x, q[1], linestyle='dashed', label='median')
            ax[k].fill_between(x, q[0], q[2], alpha=0.1) # Q1 and Q3
            # ax.legend(loc='upper right')

            maxplot = ax[k].inset_axes([0.65, 0.85, 0.3, 0.1])
            maxs = log_k.max(axis=1)
            maxplot.plot(x, maxs, color='red')
            maxplot.set_xticks([])
            maxplot.set_yticks([maxs.min(), maxs.max()])

            minplot = ax[k].inset_axes([0.65, 0.69, 0.3, 0.1])
            mins = log_k.min(axis=1)
            minplot.plot(x, mins, color='blue')
            minplot.set_xticks([])
            minplot.set_yticks([mins.min(), mins.max()])

        ax[-1].set_xlabel('Generation')
        ax[-1].set_ylabel('Cost')

        fig.savefig(os.path.join(path, 'performance%d.png' % i), bbox_inches='tight')

        # fig, ax = plt.subplots(1)
        # fig.set_size_inches(8, 8)

        for j in range(max_nn):
            fname = os.path.join(path, '%d_%dr.tsv' % (i, j))
            cost, flag = save_img(fname)
            # ax.scatter(i, cost, color=flag)
        #
        # ax.set_xlabel('Rank')
        # ax.set_ylabel('Cost')
        # fig.savefig('results/final_elites(partial).png', bbox_inches='tight')
