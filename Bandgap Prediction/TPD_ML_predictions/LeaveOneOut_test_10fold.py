# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:39:45 2020

@author: vijay
"""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from numpy import loadtxt
import scipy as sp
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

kfold = 10
ndata = 1346
nrepeat = 1

coddata = np.array([[0 for i in range(nrepeat) ] for j in range(kfold)], dtype=np.float64)
pccdata = np.array([[0 for i in range(nrepeat) ] for j in range(kfold)], dtype=np.float64)
maedata = np.array([[0 for i in range(nrepeat) ] for j in range(kfold)], dtype=np.float64)
msedata = np.array([[0 for i in range(nrepeat) ] for j in range(kfold)], dtype=np.float64)
rmsedata = np.array([[0 for i in range(nrepeat) ] for j in range(kfold)], dtype=np.float64)

cod_med = np.array([0 for i in range(kfold)], dtype=np.float64)
pcc_med = np.array([0 for i in range(kfold)], dtype=np.float64)
mae_med = np.array([0 for i in range(kfold)], dtype=np.float64)
mse_med = np.array([0 for i in range(kfold)], dtype=np.float64)
rmse_med = np.array([0 for i in range(kfold)], dtype=np.float64)


X = loadtxt("ASorted_HOIP_F32_bg.txt", comments="#", delimiter=",", unpack=False)
y = loadtxt("ASorted_HOIP_bg.txt", comments="#", delimiter=",", unpack=False)

x1 = X[0::10]
x2 = X[1::10]
x3 = X[2::10]
x4 = X[3::10]
x5 = X[4::10]
x6 = X[5::10]
x7 = X[6::10]
x8 = X[7::10]
x9 = X[8::10]
x10= X[9::10]


y1 = y[0::10]
y2 = y[1::10]
y3 = y[2::10]
y4 = y[3::10]
y5 = y[4::10]
y6 = y[5::10]
y7 = y[6::10]
y8 = y[7::10]
y9 = y[8::10]
y10= y[9::10]

X = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
y = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])

groups = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)
logo.get_n_splits(groups=groups)  # 'groups' is always required
print(logo)

ii=0;
for train_index, test_index in logo.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_traina, X_testa = X[train_index], X[test_index]
    y_traina, y_testa = y[train_index], y[test_index]
    
    X_train = np.concatenate((X_traina[0], X_traina[1], X_traina[2], X_traina[3]))
    X_test  = X_testa[0]
    
    y_train = np.concatenate((y_traina[0], y_traina[1], y_traina[2], y_traina[3]))
    y_test  = y_testa[0]
    
    # print(X_train, X_test, y_train, y_test)
    for jj in range(nrepeat):
    # GBT
        params={'n_estimators': 20000, 'max_depth': 8, 'min_samples_split': 2,
            'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
        reg = ensemble.GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test) 

        cod = r2_score(y_test, y_pred)
        pcc = sp.stats.pearsonr(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse= sqrt(mse)
        
        coddata[ii,jj] = cod
        pccdata[ii,jj] = pcc[0]
        maedata[ii,jj] = mae
        msedata[ii,jj] = mse
        rmsedata[ii,jj] = rmse
        
# # plot feature importance     
#         feature_importance = reg.feature_importances_
#         sorted_idx = np.argsort(feature_importance)
#         sorted_arr = sorted_idx[::-1]
#         pos = np.arange(sorted_idx.shape[0]) + .5
#         fig = plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.barh(pos, feature_importance[sorted_idx], align='center')
#         plt.yticks(pos, sorted_idx)
#         plt.title('Feature Importance (MDI)')       

    ii=ii+1

for kk in range(kfold):
    cod_med[kk] = np.median(coddata, axis=1)[kk]
    print("CODmed :", cod_med[kk])
    pcc_med[kk] = np.median(pccdata, axis=1)[kk]
    print("PCCmed :", pcc_med[kk])
    mae_med[kk] = np.median(maedata, axis=1)[kk]
    print("MAEmed :", mae_med[kk])
    mse_med[kk] = np.median(msedata, axis=1)[kk]
    print("MSEmed :", mse_med[kk])
    rmse_med[kk] = np.median(rmsedata, axis=1)[kk]
    print("RMSEmed :", rmse_med[kk])
 
foutname = "hoip_EPstats_F32_10fold.mat"        
sio.savemat(foutname, {"CODmed": cod_med, "PCCmed": pcc_med, "MAEmed": mae_med, "MSEmed": mse_med, "RMSEmed": rmse_med})    
        

    
     