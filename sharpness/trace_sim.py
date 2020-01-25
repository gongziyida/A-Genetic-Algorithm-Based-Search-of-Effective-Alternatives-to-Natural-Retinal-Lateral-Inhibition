#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

with open('results/TRACE', 'r') as log:
    lines = log.readlines()

n = int(lines[0])

states = [None] * n

for i in range(n):
    states[i] = [list(map(float, lines[l].split(' ')[:-1])) for l in range(i+1, len(lines), n)]

fig, ax = plt.subplots(n)
fig.tight_layout()

for i in range(n):
    ax[i].plot(np.array(states[i]))

plt.show()
