  ![](<./stock_prediction.jpg>)
# Stock Return Prediction Research

## Table of Contents

- [Introduction](#introduction)
- [Part 1: How the Data Looks Like (Descriptive Analysis and Outliers)](#part-1-how-the-data-looks-like-descriptive-analysis-and-outliers)
- [Part 2: How to Deal with Missing Data Imputation](#part-2-how-to-deal-with-missing-data-imputation)
- [Part 3: How the Performance of Back Testing from Both Linear and Non-Linear Models](#part-3-how-the-performance-of-back-testing-from-both-linear-and-non-linear-models)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Introduction

This post delves into the intricacies of stock return prediction, outlining various aspects of data analysis, modeling, and back-testing. We'll cover data structure introduction, descriptive analysis, missing value imputation, and the performance of both linear and non-linear models in the context of back-testing.

## Part 1: How the Data Looks Like (Descriptive Analysis and Outliers)

### Data Structure Introduction

- **Y:** Represents the vector of stock returns.
- **X:** Comprises an 8-column matrix, with each column representing a distinct stock feature.

### Descriptive Analysis

#### Y:

- Y's skewness and kurtosis indicate significant deviations when all data is included.
- A box plot of Y highlights the presence of outliers.
- To address this, fat-tail distribution techniques were applied, including the removal of data larger than the top 0.1% and smaller than the bottom 0.1%.

  ![](<./Box_plot_of_Y.jpg>)
  ![](<./Y_remove_outlier.jpg>)
  
#### X:

- Features in X exhibit improved skewness and kurtosis compared to Y.
- Some features have high kurtosis, but they are retained since feature selection methods will be employed.
- Notably, data rows removed from Y are also removed from X.
- A box plot of X is used to visualize the distribution of features.

![](<./Des_X.JPG>)

#### Covariance Matrix:

- No significant linear relationship is observed between any two features, except for X1 and X2. Multicollinearity is not a primary concern, as stepwise selection methods will address this.

![](<./CovX.JPG>)

## Part 2: How to Deal with Missing Data Imputation

- The number of missing observations for each feature is depicted in a table.
- K-nearest neighbor (KNN) imputation with an expectation-maximization (EM) algorithm is employed. This is because stocks can form clusters based on features.
- The imputation process involves using column means, grouping data into clusters using KNN, and iteratively updating cluster centers until stability is achieved or the maximum iteration limit is reached.

## Part 3: How the Performance of Back Testing from Both Linear and Non-Linear Models

- Stepwise forward feature selection methods based on adjusted R-square or user-defined score measures are used for all models.
- Two linear models are tested: classic linear regression with OLS and Elastic Net (EN) to harness L1 and L2 regularization.
- For the non-linear approach, Support Vector Machine Regression (SVM) and Multilayer Perceptron (MP) models are considered.

### Model Back-Testing

- A moving window out-sample back-testing method is employed to evaluate model performance over various time periods.
- This method allows for model adaptation to changing coefficients as time periods shift.
  
![](<./Model_r_square.jpg>)

## Conclusion

- In the context of in-sample feature selection based on adjusted R-square, the performance of the models is evaluated.
- Linear models, especially OLS, outperform non-linear models in terms of R-square.
- The table provides fLoss, R-square, R-value, and correlation results for each model, offering insights into their effectiveness in predicting stock returns.


## Acknowledgments

- [Support Vector Machine - Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [ElasticNet - scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [SVR (Support Vector Regression) - scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [MLPRegressor - scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)
- Rishi K. Narang, "Inside the Black Box-The Simple Truth About Quantitative Trading." John Wiley & Sons, Inc, 2009: 11-38.
