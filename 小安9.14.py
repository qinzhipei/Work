# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:43:52 2023

@author: admin
"""

import numpy as np
x = np.array(eval(input('Enter X:')))
m = np.array(eval(input('Enter M:')))

n = x.shape[0]
k = m.shape[0]
d = x.shape[1]

D = []
for i in range(n):
    for i in range(k):
        d.append((x[i][0] - m[0][0])**2 + (x[i][1] - m[0][1])**2)
        d1 = (x[i][0] - m[1][0])**2 + (x[i][1] - m[1][1])**2
    if d0 < d1:
        print('{} -> {}'.format(x[i],1))
    if d0 > d1:
        print('{} -> {}'.format(x[i],2))
        

x = np.array([[1,1],[3,1],[3,3],[1,3]])
m = np.array([[1,2],[3,2]])