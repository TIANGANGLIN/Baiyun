# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:19:37 2020

@author: nihao
"""

import pickle
import pandas as pd
import  numpy as np
import math
from sklearn import preprocessing, model_selection,linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
# from neural_network import MLPRegressor_cuda

from sklearn.pipeline import make_pipeline

# l1-true,l2-false
def RMSE(l1,l2):
    length = len(l1)
    sum = 0
    for i in range(length):
        sum = sum + np.square(l1[i]-l2[i])
    return math.sqrt(sum/length)
#define MAE
def MAE(l1,l2):
    n = len(l1)
    l1 = np.array(l1)
    l2 = np.array(l2)
    mae = sum(np.abs(l1-l2))/n
    return mae
#def MAPE
def MAPE(l1,l2):
    n = len(l1)
    l1 = np.array(l1)
    l2 = np.array(l2)
    for i in range(len(l1)):
        if l1[i] == 0:
            l1[i] = 0.01
    mape = sum(np.abs((l1-l2)/l1))/n
    return mape


root = './'

with open(root+'x_train_sta.pkl','rb') as file:
    x_train_sta = pickle.load(file)
    
with open(root+'x_test_sta.pkl','rb') as file:
    x_test_sta = pickle.load(file)
    
with open(root+'y_train_sta.pkl','rb') as file:
    y_train_sta = pickle.load(file)

with open(root+'y_test_sta.pkl','rb') as file:
    y_test_sta = pickle.load(file)


#多层感知机回归
param_grid_12 = {'activation':['identity','logistic','tanh','relu'], 'solver':['lbfgs','sgd','adam'],
                 'alpha':[0.00001,0.0001,0.001],'learning_rate':['constant','invscaling','adaptive']}
reg_12 = MLPRegressor()
# reg_12 = MLPRegressor_cuda()
grid_search_12 = GridSearchCV(reg_12, param_grid_12, cv=5)
grid_search_12.fit(x_train_sta, y_train_sta[:,0])

print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_12.best_params_)
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_12.score(x_train_sta, y_train_sta))
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_12.best_score_)  

y_pre_12 = grid_search_12.predict(x_test_sta)
with open(root+'y_pre_12.pkl','wb') as file:
    pickle.dump(y_pre_12,file)

y_pre_12 = y_pre_12.reshape(len(y_pre_12),1)       
print('RMSE',RMSE(y_test_sta,y_pre_12))
print('MAE',MAE(y_test_sta,y_pre_12))
print('MAPE',MAPE(y_test_sta,y_pre_12))


