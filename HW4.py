# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:48:53 2020

@author: Lucia
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pylab
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection
from sklearn.linear_model import RidgeCV

df = pd.read_csv('D:\Desktop\IE517 ML in Fin Lab\Module4\HW4\housing.csv')


#Part 1: Exploratory Data Analysis

#heatmap
sns.heatmap(df.corr(), annot = True, annot_kws = {"size": 7})
plt.yticks(rotation = 0, size = 14)
plt.xticks(rotation = 90, size = 14)  # fix ticklabel directions and size
plt.tight_layout()
plt.show()
#scatterplot matrix
sns.pairplot(df, height = 2.5)
plt.tight_layout()
plt.show()
#Box plot
sns.boxplot(data = df)
plt.xlabel('Attribute')
plt.ylabel('Quantile Ranges')
plt.title("Box plot")
plt.xticks(rotation = 90)
plt.show()

#Split data into training and test sets
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
sc = StandardScaler()
#X_train_std = sc.transform(X_train)
#X_test_std = sc.transform(X_test)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


#Part 2: Linear regression


#Fit a linear model using SKlearn to all of the features of the dataset
slr = LinearRegression()
slr.fit(X_train_std, y_train)
y_train_pred = slr.predict(X_train_std)
y_test_pred = slr.predict(X_test_std)

#plot the residual errors
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values (MEDV)')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

#Describe the model: coefficients and y intercept
print('coef:')
print(pd.DataFrame(slr.coef_,df.columns[:-1]))
print('    ')
print('intercept:')
print(slr.intercept_)

#calculate performance metrics: MSE and R2
slr_mse = mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)
slr_r2 = r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)
print('MSE train: %.3f, test: %.3f' % slr_mse)
print('R^2 train: %.3f, test: %.3f' % slr_r2)


##Part 3.1: Ridge regression

#Test some settings for alpha
ridge_alpha_space = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
ridge_r2_train = []
ridge_r2_test = []
ridge_mse_train = []
ridge_mse_test = []

ridge = Ridge(normalize = True)

for alpha in ridge_alpha_space:
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    ridge_r2_train.append(r2_score(y_train, y_train_pred))
    ridge_r2_test.append(r2_score(y_test, y_test_pred))
    ridge_mse_train.append(mean_squared_error(y_train, y_train_pred))
    ridge_mse_test.append(mean_squared_error(y_test, y_test_pred))

ridge_best_mse = min(ridge_mse_test)
ridge_best_alpha = ridge_alpha_space[ridge_mse_test.index(ridge_best_mse)]
print('The best alpha:')
print(ridge_best_alpha)

#find the best alpha use cv
ridgecv = RidgeCV(alphas = ridge_alpha_space, cv = 5)
ridgecv.fit(X_train,y_train)
ridge_best_alpha = ridgecv.alpha_
print("The best alpha is:", ridge_best_alpha)

#Fit a Ridge model using the best alpha
ridge_best = Ridge(alpha = ridge_best_alpha,normalize = True)
ridge_best.fit(X_train, y_train)
ridge_y_train_pred = ridge_best.predict(X_train)
ridge_y_test_pred = ridge_best.predict(X_test)

#Describe the model: coefficients and y intercept
print('coef:')
print(pd.DataFrame(ridge_best.coef_,df.columns[:-1]))
print('    ')
print('intercept:')
print(ridge_best.intercept_)

#plot the residual errors
plt.scatter(ridge_y_train_pred, ridge_y_train_pred-y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(ridge_y_test_pred, ridge_y_test_pred-y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values (MEDV)')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, colors='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#calculate performance metrics: MSE and R2
ridge_mse = mean_squared_error(y_train, ridge_y_train_pred), mean_squared_error(y_test, ridge_y_test_pred)
ridge_r2 = r2_score(y_train, ridge_y_train_pred), r2_score(y_test, ridge_y_test_pred)
print('MSE train: %.3f, test: %.3f' % ridge_mse)
print('R^2 train: %.3f, test: %.3f' % ridge_r2)


#Part 3.2: LASSO regression

#Test some settings for alpha
lasso_alpha_space = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
lasso_r2_train = []
lasso_r2_test = []
lasso_mse_train = []
lasso_mse_test = []

lasso = Lasso(normalize = True)

for alpha in lasso_alpha_space:
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    lasso_r2_train.append(r2_score(y_train, y_train_pred))
    lasso_r2_test.append(r2_score(y_test, y_test_pred))
    lasso_mse_train.append(mean_squared_error(y_train, y_train_pred))
    lasso_mse_test.append(mean_squared_error(y_test, y_test_pred))

lasso_best_mse = min(lasso_mse_test)
lasso_best_alpha = lasso_alpha_space[lasso_mse_test.index(lasso_best_mse)]
print('The best alpha:')
print(lasso_best_alpha)

#find the best alpha use cv
lassocv = LassoCV(alphas = lasso_alpha_space, cv = 5)
lassocv.fit(X_train,y_train)
lasso_best_alpha = lassocv.alpha_
print("The best alpha is:", lasso_best_alpha)

#Fit a Ridge model using the best alpha
lasso_best = Lasso(alpha=lasso_best_alpha,normalize = True)
lasso_best.fit(X_train, y_train)
lasso_y_train_pred = lasso_best.predict(X_train)
lasso_y_test_pred = lasso_best.predict(X_test)

#Describe the model: coefficients and y intercept
print('coef:')
print(pd.DataFrame(lasso_best.coef_,df.columns[:-1]))
print('    ')
print('intercept:')
print(lasso_best.intercept_)

#plot the residual errors
plt.scatter(lasso_y_train_pred, lasso_y_train_pred-y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(lasso_y_test_pred, lasso_y_test_pred-y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values (MEDV)')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, colors='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#calculate performance metrics: MSE and R2
lasso_mse = mean_squared_error(y_train, lasso_y_train_pred), mean_squared_error(y_test, lasso_y_test_pred)
lasso_r2 = r2_score(y_train, lasso_y_train_pred), r2_score(y_test, lasso_y_test_pred)
print('MSE train: %.3f, test: %.3f' % lasso_mse)
print('R^2 train: %.3f, test: %.3f' % lasso_r2)


print("-----------")
print("My name is Lu Yuan")
print("My NetID is: luyuan3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")