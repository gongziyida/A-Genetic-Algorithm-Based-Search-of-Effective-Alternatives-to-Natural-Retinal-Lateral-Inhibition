#! /usr/bin/env python3

COMMON = 'red'
NO_INTERNEURON = 'black'
NO_NON_TRIVIAL_W = 'gray'

def save_img(fname):
    i = 2
    counter = 0
    n_cells = {}
    connections = {}

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

    for i, (s, t) in enumerate(connections.keys()):
        g = nx.MultiDiGraph()

        n = n_cells[s]
        m = n_cells[t]
        c = connections[(s, t)]

        for p in range(m):
            for q in range(n):
                if np.abs(c[p, q]) > 0.01:
                    g.add_edge((s, q), (t, p), color=c[p, q])

        # Extract a list of edge colors
        edges = g.edges()
        edge_colors = [g[u][v][0]['color'] for u, v in edges]

        # define fixed positions of nodes
        nodes = g.nodes()
        fixed_pos = {}
        for n in nodes:
            fixed_pos.update({n: (n[0], (n[1] - n_cells[n[0]]/2) * 2)})

        # get the nx positions
        pos = nx.spring_layout(g, pos=fixed_pos, fixed=fixed_pos.keys())

        # draw
        nx.draw(g, pos=pos, node_size=50, node_color='black',
                edge_cmap=plt.get_cmap('bwr'), edge_color=edge_colors, edge_vmin=-1, edge_vmax=1,
                ax=ax[i])

        ax[i].text(0.5, 1, '%d -> %d' % (s, t), transform=ax[i].transAxes, fontsize=10)

    fig.savefig(fname, dpi=200, bbox_inches='tight')

    return float(lines[0]), COMMON

if __name__ == '__main__':
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    log = np.loadtxt('results/LOG', dtype=np.float64)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    generations = np.arange(log.shape[0])
    # for i in range(log.shape[0]):
        # ax.scatter(generations, log[:, i], s=1, c='b')
    ax.boxplot(log.T, sym='+')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cost')
    fig.savefig('results/performance.png', bbox_inches='tight')

    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)

    fname = 'results/%d'
    for i in range(int(sys.argv[1])):
        cost, flag = save_img(fname % i)
        ax.scatter(i, cost, color=flag)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Cost')
    fig.savefig('results/final_elites.png', bbox_inches='tight')
