#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function


import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt


print('Loading data...')
datafit = sio.loadmat('datafit.mat')
X = datafit['X']
Y = np.squeeze(datafit['Y'])
#Y = datafit['Y']

print('Loaded X matrix shape:', X.shape)
print('Loaded Y vector shape:', Y.shape)

mCovX = np.corrcoef(X.T)

print('Y describe', stats.describe(Y, axis=0))
nGlobalMin = np.amin(Y) - 0.001
print('Y describe', stats.describe(np.log(Y - nGlobalMin), axis=0))
nQu = np.percentile(Y, 0.1)
nQl = np.percentile(Y,99.9)
index = np.where((Y>nQu) * (Y<nQl))
Y = Y[index]
X = X[index]
dR = stats.describe(Y, axis=0)
mDesarray = np.array([[dR.mean, dR.variance, dR.skewness, dR.kurtosis]])

for iCol in range(8):
    Xp = X[:, iCol]
    Xp = Xp[~np.isnan(Xp)]

    plt.boxplot(Xp)
    plt.show()
    print('Col num %d' %iCol, stats.describe(Xp, axis=0))
    dR = stats.describe(Xp, axis=0)
    #mDesarray = np.concatenate((mDesarray, np.array([[dR.mean, dR.variance, dR.skewness, dR.kurtosis]])), axis=0)
np.savetxt('Description.csv', mDesarray, delimiter=',')