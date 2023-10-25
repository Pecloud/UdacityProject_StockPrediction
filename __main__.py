
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import csv
import numpy as np
import scipy.io as sio
from scipy import stats
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import KMean_MissingImpute as KMI
import ModelEva

def main():

    print('Loading data...')
    datafit = sio.loadmat('datafit.mat')
    X = datafit['X']
    Y = np.squeeze(datafit['Y'])


    print('Loaded X matrix shape:', X.shape)
    print('Loaded Y vector shape:', Y.shape)

    print('Number NaN values for each column after imputation')
    print(np.sum(np.isnan(X) , axis = 0) )#Check mssing number of each column

    #Removing extreme values
    nQl = np.percentile(Y, 0.1)
    nQu = np.percentile(Y, 99.9)
    index = np.where((Y>=nQl) * (Y<=nQu))
    Y = Y[index]
    X = X[index]


    #Set model arguments
    sModelUse = 'EN'
    sFileName = 'flsqS1000_200'
    nTrainWindowL = 1000 #Influences the running time and preceiceness of each loop
    nTestWindowL = 200 #Influences how many loops
    nTestStart = nTrainWindowL #This is the first date in the current test window

    mResultIn = np.empty([1,4])
    mResult = np.empty([1,4])
    mResult = np.delete(mResult, 0, 0)
    mResultIn = np.delete(mResultIn, 0, 0)
    lFeatures = []

    #Moving window back testing
    while(nTestStart <= X.shape[0]):
        mXtrain = X[nTestStart - nTrainWindowL : nTestStart, :]
        mYtrain = Y[nTestStart - nTrainWindowL : nTestStart]
        print('Imputing missing values by k-means clustering')
        mXtrain = KMI.kmeans_missing(mXtrain, 10)[2]  # Impute missing data by k-means clustering

        nTestEnd = nTestStart + nTestWindowL
        nTestEnd = nTestEnd if nTestEnd <= X.shape[0] else X.shape[0] #If it reaches the end of the data
        mXtest = X[nTestStart : nTestEnd, :]
        mYtest = Y[nTestStart : nTestEnd]
        mXtest = KMI.kmeans_missing(mXtest, 10)[2]  # Impute missing data by k-means clustering




        #Transformation trail. The results are not good
        nGlobalMin = np.amin(mXtrain) - 0.001
    #    mXtrain = np.concatenate((mXtrain, 1 / (mXtrain - nGlobalMin)), axis=1)
    #    mXtest = np.concatenate((mXtest, 1 / (mXtest - nGlobalMin) ), axis=1)
        mXtrain = np.concatenate( (mXtrain, mXtrain**2, 1/(mXtrain - nGlobalMin) ), axis=1)
        mXtest = np.concatenate((mXtest, mXtest**2, 1 / (mXtest - nGlobalMin) ), axis=1)

    #    mXtrain = np.log(mXtrain - nGlobalMin)
    #    mXtest = np.log(mXtest - nGlobalMin)


    #    mYtrain = 3 * mXtrain[:, 0] * 3 * mXtrain[:, 1] + 6 * mXtrain[:, 4]  #Only for test purpose
    #    mXtrain[:,1] = mYtrain Only for test purpose

        #####
        # Model here
        print('Featu1res selection...')
        model, lFeaS,nScore = ModelEva.fForwardSelectFeature(mXtrain, mYtrain, sScore='fLoss', nAggre = 2, nConserv = 1, sModel = sModelUse)
        lFeaS = sorted(lFeaS)
        print('Chose features: ',lFeaS, '\n', 'Score: ',nScore, '\n' )
        mXtrain = mXtrain[:, lFeaS]
        mXtest = mXtest[:, lFeaS]
        print('Xshape',mXtrain.shape, mXtest.shape )
        print('Building and training model...')

        model.fit(mXtrain, mYtrain)
        mY_hat = model.predict(mXtrain)
        mY_pred = model.predict(mXtest)

        print('Evaluate in-sample model...')
        nLoss = ModelEva.fLoss(mYtrain, mY_hat, nAggre=2, nConserv=1)
        r2 = r2_score(mYtrain, mY_hat)
        r_value = np.sign(r2) * np.sqrt(abs(r2))
        correlation = np.corrcoef(mYtrain, mY_hat)[0, 1]
        mResultIn = np.concatenate((mResultIn, np.array([[nLoss, r2, r_value, correlation]])), axis=0)
        print('Evaluating in-sample result', mResultIn[-1, :])

        print('Evaluating model...')
        nLoss = ModelEva.fLoss(mYtest, mY_pred, nAggre=2, nConserv=1)
        r2 = r2_score(mYtest, mY_pred)
        r_value = np.sign(r2) * np.sqrt(abs(r2))
        correlation = np.corrcoef(mYtest, mY_pred)[0, 1]
        mResult = np.concatenate((mResult, np.array([[nLoss, r2, r_value, correlation]])), axis=0)
        print('Evaluating result',mResult[-1, :])

        #####
        #Save results for further analysis
        lFeatures = lFeatures.insert(-1, lFeaS)
        with open(sFileName + sModelUse + 'SelectFeatures.csv', 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(lFeatures)

        np.savetxt(sFileName + sModelUse + 'Insamp.csv', mResultIn, delimiter=',')
        np.savetxt(sFileName + sModelUse + 'Outsamp.csv', mResult, delimiter=',')
        nTestStart = nTestEnd
if __name__ == "__main__":
    main()