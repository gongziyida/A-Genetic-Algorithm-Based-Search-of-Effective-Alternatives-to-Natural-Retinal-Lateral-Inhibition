import numpy as np
import matplotlib.pyplot as plt
import re

w_rm = re.compile(r'(?<!^)[ \s]+')                                                                

with open('test.txt', 'r') as f:
    y = f.readlines()

n = int(y[0])
y = y[1:]

ys = [[y[j] for j in range(i, len(y), n)] for i in range(n)]                                      

for i in range(n): 
    for j in range(200): 
        ysp = w_rm.split(ys[i][j])[:-1] 
        try: 
            ys[i][j] = list(map(float, ysp)) 
        except: 
            ys[i][j] = list(map(float, ysp[1:])) 

for i in range(n): 
    ys[i] = np.array(ys[i]) 

fig, ax = plt.subplots(1, n) 
fig.tight_layout()
for i in range(n): 
    im = ax[i].imshow(ys[i], aspect='auto')
    fig.colorbar(im, ax=ax[i])
plt.show()
