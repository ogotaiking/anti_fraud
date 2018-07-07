import os

import numpy as np
import pandas as pd
import tensorflow as tf
import copy as copy

import atecml.data
import random

from contextlib import contextmanager
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA

#build Models...
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from lightgbm.sklearn import LGBMModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE, ADASYN
import copy as copy
import random


def model_train(train_df,feature_list,model_name):
    model_cache_name = './'+model_name+'.model'
    temp = model_name.split('__')
    model_type = temp[0]
    target = temp[1]
    f_idx = int(temp[3])
    
    select_feature = feature_list[f_idx]
    categorical=[]
    for item in select_feature:
        if (item in atecml.data.CATE_FEATURE_LIST):
            categorical.append(item)

    lgbm_params = {
        'boosting_type': model_type,
        'objective': 'regression',
        'use_missing' : 'true',
        'learning_rate': 0.05,
        'num_leaves': 64,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 600,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'colsample_bytree': 0.9,
        'subsample': 0.85,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'min_child_weight': 0.05,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.01,  # L1 regularization term on weights
        'reg_lambda': 0.1,  # L2 regularization term on weights
        'nthread': 40,
        'verbose': 0,
        'n_estimators' : 400,
        'metric':{'l2', 'auc'}
    }

    if (target == 'Normal'):
        params = {
            'scale_pos_weight' : 0.01
        }
    else:
        params = {
            'scale_pos_weight' : 99
        }
    lgbm_params.update(params)

    with atecml.data.timer('> {} <: Training...'.format(model_name)):
        clr = LGBMModel(**lgbm_params)
        clr.fit(train_df[select_feature],train_df[target],verbose=50,categorical_feature=categorical)
    joblib.dump(clr,model_cache_name)
    return clr


#Loading Data....
data = atecml.data.load_train()
train_df = atecml.data.filter_date(data,start_date='2017-09-05',end_date='2017-10-15')
val_df = atecml.data.filter_date(data,start_date='2017-10-16',end_date='2018-10-15')

predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_SUM]

feature_tree_num = 5
if (os.path.exists('./feature_list.dat')):
    print('Load Feature List from persistant store...')
    feature_list = joblib.load('./feature_list.dat')
else:
    print('Generate Random Feature List...')
    feature_list = {}
    for idx in range(0,feature_tree_num):
        feature_set = set(random.sample(predictors,120))
        feature_list[idx] = list(feature_set)
    joblib.dump(feature_list,'./feature_list.dat')

train_model =[]
for idx in range(0,1):
    for item in ['dart','gbdt','rf' ]:
        for feature_grp_idx in range(0,feature_tree_num):
            for target in ['Normal','Fraud']:
                train_id = item + '__'+target +'__'+str(idx) +'__' + str(feature_grp_idx)
                train_model.append(train_id)

trained_model_list =[]
with atecml.data.timer('Classification: Model Training'):
    for train_id in tqdm(range(len(train_model))):
        fit_model = model_train(train_df,feature_list,train_model[train_id])
        trained_model_list.append(fit_model)

