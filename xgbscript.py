import pandas as pd
import numpy as np
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

def get_timespan(df, num_days, max_days,future_date):
    feature = []
    for i in range(max_days,len(df)-future_date):
        temp_sum = 0
        for j in range(i-num_days,i):
            temp_sum+=df['Price'][j]
        feature.append(temp_sum/num_days)
    return np.array(feature)

def get_dow(df,num_days,max_days,future_date):
    feature = []
    for i in range(max_days, len(df) - future_date):
        temp_sum = 0
        for j in range(i - num_days*7, i,7):
            temp_sum += df['Price'][j]
        feature.append(temp_sum / num_days)
    return np.array(feature)
from sklearn.metrics import mean_squared_error
df = pd.read_csv('a.csv')
X = pd.DataFrame({
    "day_1": get_timespan(df,1,140,30),
    "mean_3": get_timespan(df,3,140,30),
    "mean_7": get_timespan(df,7,140,30),
    "mean_14": get_timespan(df,14,140,30),
    "mean_30": get_timespan(df,30,140,30),
    "mean_60": get_timespan(df,60,140,30),
    "mean_140": get_timespan(df,140,140,30),
    "mean_4_dow": get_dow(df,4,140,30),
    "mean_20_dow": get_dow(df,20,140,30)})
y = df['Price'][170:]
#print(X.shape, y.shape)

param = {}
param['objective'] = 'reg:linear'
param['eta'] = 0.5
param['max_depth'] = 3
param['silent'] = 1
param['eval_metric'] = 'rmse'
param['min_child_weight'] = 4
param['subsample'] = 0.8
param['colsample_bytree'] = 0.7
param['seed'] = 137
num_rounds = 100

dtrain = xgb.DMatrix(X,label=y)