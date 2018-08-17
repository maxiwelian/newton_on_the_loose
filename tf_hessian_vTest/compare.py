

import numpy as np

with open('weight.csv', 'r') as f:
    x = np.loadtxt(f, delimiter=',')

with open('weight_bishop.csv', 'r') as f:
    y = np.loadtxt(f, delimiter=',')

z = x - y

for i in range(24):
    for j in range(24):
        if z[i,j] > 10**-8:
            print('Error')