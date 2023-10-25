# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

def fLoss(vTure, vPredict, nAggre = 2, nConserv = 1):
    #vTure: true return vector
    #vPredict: Predicted return vector
    #nAggre: The penalty coefficient if the prediction is aggressive
    #nConserv: The penalty coefficient if the prediction is conservative

    #Returns the mean loss score of the prediction. This value is always non-nagetive

    vPercentErr = (vTure - vPredict) #/ (vTure) #For percentage bias
    iIndex = vPercentErr <= 0 #The true return is less than predicted return. So our prediction is too aggresive
    vPercentErr[iIndex] *= -nAggre
    iIndex = vPercentErr > 0
    vPercentErr[iIndex] *= nConserv

    return np.mean(vPercentErr)

def fModelSelection(sModel = 'Linear'):
    #sModel: Choose different model

    #Return model object

    if sModel == 'Linear':# Linear model for OLS
        model = linear_model.LinearRegression()
    elif sModel == 'SVR': #SVM regressing
        dParameters = {'kernel':['linear','rbf'],'C':[3,5,9], 'gamma':[0.03,0.05,0.1]} # 'poly' 'degree':[2,3,4],
        model = GridSearchCV(SVR(), dParameters)
    elif sModel == 'EN': # Elastic Net
        dParameters = {'alpha': [1, 1.5, 2], 'l1_ratio' : [0.3, 0.5, 0.7]}
        model = GridSearchCV(ElasticNet(random_state=0), dParameters)
    elif sModel == 'MLP': # Multilayer perceptron
        dParameters = {'alpha': [0.0001,0.001, 0.01, 0.1]}
        model = GridSearchCV(MLPRegressor(), dParameters)
    else:#Exception handeling
        raise ValueError('Invalide model argument.for sModel! Your sModel is: %s' %sModel)
    return model


def fForwardSelectFeature(mX,vY, sScore = 'adj_r2', nAggre = 2, nConserv = 1, sModel = 'Linear'):
    #X: numpy array (matrix), independent variables with one column for one feature
    #Y: numpy array (vector), dependent variables
    #sScore: The criteria for selection, the higher the better
    #    fLoss: Loss function I defined above
    #    adj_r2: Adjusted R square

    #Returns (current model, a list shows the feature# (column#) selected by the forwarding method, and the finial score)

    lSelectedFea = []
    lCandidateFea = list(range(mX.shape[1]))
    nCurrentScore = -1000000.0
    nBestScore = -1000000.0

    model = fModelSelection(sModel)
    """
    """
    #In  we do not want the loop to select features
    if sModel == ' ':
        model.fit(mX, vY)#This step is really slow
        #print(model.best_estimator_.coef_)#For grid search
        #print(model.coefs_)#For MLP

        vYhat = model.predict(mX)
        print('Parameters grid search')
        if sScore == 'fLoss':
            nScore = -fLoss(vY, vYhat, nAggre, nConserv)  # The highr the better. So negative
        elif sScore == 'adj_r2':
            nScore = r2_score(vY, vYhat)
            print('r2: ', nScore)
            # nScore = model.score(mXpart, vY)
            nScore = 1.0 - ((1.0 - nScore) * (mX.shape[0] - 1.0) / ( mX.shape[0] - len(lCandidateFea) - 1.0))  # Adj R2 = 1 - (1 - R2) * (n-1)/n-k-1
            print('adj_r2: ', nScore)
        else:  # Exception handeling
            raise ValueError('Invalide model argument.for sScore! Your sScore is: %s' % sScore)
        return model, lCandidateFea, nScore


    while lCandidateFea and nCurrentScore == nBestScore:
        lCandidateScores = []
        for nCandidateFea in lCandidateFea:
            lFeatures = lSelectedFea + [nCandidateFea]
            mXpart = np.copy(mX[:,lFeatures ])
            model.fit(mXpart, vY)
            vYhat = model.predict(mXpart)
            #print('vYhat shape: ', vYhat.shape)

            if sScore == 'fLoss':
                nScore = -fLoss(vY, vYhat, nAggre, nConserv) #The highr the better. So negative
            elif sScore == 'adj_r2':
                nScore = r2_score(vY, vYhat)
                #nScore = model.score(mXpart, vY)
                nScore = 1.0 - ( (1.0 - nScore) * (mXpart.shape[0] - 1.0) / (mXpart.shape[0] - len(lFeatures)- 1.0) ) #Adj R2 = 1 - (1 - R2) * (n-1)/n-k-1
            else:#Exception handeling
                raise ValueError('Invalide model argument.for sScore! Your sScore is: %s' %sScore)

            lCandidateScores.append((nScore, nCandidateFea))

        lCandidateScores.sort(reverse=True)
        nBestScore, nBestCandidateFea = lCandidateScores[0]#Choose the highest score and feature
        if nCurrentScore < nBestScore:
            lCandidateFea.remove(nBestCandidateFea)
            lSelectedFea.append(nBestCandidateFea)
            nCurrentScore = nBestScore



    return model, lSelectedFea, nCurrentScore
