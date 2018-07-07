import seaborn as sns
import os

import numpy as np
import pandas as pd
import atecml.data
from tqdm import tqdm

#训练集为第一步build的纬度提升矩阵，并过滤掉unknown标签
train_df = pd.read_pickle('./01_train.dat')
val_df =  pd.read_pickle('./01_test.dat')

#train_df = atecml.data.load_train()
#val_df = atecml.data.load_test()
train_df.loc[train_df.label == 0, 'Fraud'] = 0
train_df.loc[train_df.label != 0, 'Fraud'] = 1

import joblib
#predictors =joblib.load('./woe_feature.dat')
predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_COLUMNS]


target = 'Fraud'
DateFold={}

DateFold[0] = set(atecml.data.filter_date(train_df,start_date='2017-09-05',end_date='2017-09-12').index)
DateFold[1] = set(atecml.data.filter_date(train_df,start_date='2017-09-13',end_date='2017-09-20').index)
DateFold[2] = set(atecml.data.filter_date(train_df,start_date='2017-09-21',end_date='2017-09-28').index)
DateFold[3] = set(atecml.data.filter_date(train_df,start_date='2017-09-29',end_date='2017-10-06').index)
DateFold[4] = set(atecml.data.filter_date(train_df,start_date='2017-10-07',end_date='2017-10-14').index)
DateFold[5] = list(atecml.data.filter_date(train_df,start_date='2017-10-15',end_date='2017-11-24').index)

all_list = set(train_df.index) - set(DateFold[5])
len(all_list),len(DateFold[5])

X = np.array(train_df[predictors])
Y = np.array(train_df[target])

val_index = DateFold[5] 
#train_index = list(all_list)
train_index = list(DateFold[3]) + list(DateFold[4])

X_train = X[train_index]
y_train = Y[train_index]

X_val = X[val_index]
y_val = Y[val_index]
    
X_test = np.array(val_df[predictors])

print(X_train.shape, y_train.shape)

import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)

print('设置参数')
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_logloss',
          'verbose' : -1,
          }

### 交叉验证(调参)
print('交叉验证')
min_merror = float('Inf')
best_params = {}

# 准确率
print("调参1：提高准确率")
for num_leaves in range(20,200,10):
    for max_depth in range(3,10,3):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
        print('---------------------------------------------------')
        print('L:',num_leaves,'D:',max_depth)
        print('---------------------------------------------------')

        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            num_boost_round=500,
                            seed=2018,
                            nfold=5,
                            metrics=['auc'],
                            early_stopping_rounds=10,
                            verbose_eval=20
                            )
            
        mean_merror = pd.Series(cv_results['auc-mean']).min()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmin()
            
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
            print('Best:',num_leaves,max_depth)
            
params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

print('Final Best:',num_leaves,max_depth)
joblib.dump(best_params,'./best_leave_depth.dat')



