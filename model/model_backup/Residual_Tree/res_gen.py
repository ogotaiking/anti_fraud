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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE, ADASYN
import copy as copy
import random

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='binary_error',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'dart',
        #'drop_rate' : 0.2, #Dart Only, DropoutRate
        'objective': objective,
        'use_missing' : 'true',
        'learning_rate': 0.15,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 64,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 600,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'feature_fraction': 0.4,
        'subsample': 0.85,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'min_child_weight': 0.05,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 2000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 5,  # L1 regularization term on weights
        'reg_lambda': 10,  # L2 regularization term on weights
        'nthread': 40,
        'verbose': -1,
        'scale_pos_weight' : 0.01,
        'metric':metrics
    }

    lgb_params.update(params)

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval, 
                     feval=feval)

    return bst1

def model_train(train_df,val_df,feature_list,model_name):
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

    if (target == 'Normal'):
        params = {
            'scale_pos_weight' : 0.01,
            'boosting_type': model_type
        } 
    else:
        params = {
            'scale_pos_weight' : 99,
            'boosting_type': model_type
        }

    with atecml.data.timer('> {} <: Training...'.format(model_name)):
        bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        select_feature, 
                        target, 
                        objective='binary', 
                        metrics='binary',
                        early_stopping_rounds=100, 
                        verbose_eval=50, 
                        num_boost_round=5000, 
                        categorical_features=categorical)
    joblib.dump(bst,model_cache_name)
    return bst


#Loading Data....
data = pd.read_pickle('./res.dat')
train_df = atecml.data.filter_date(data,start_date='2017-09-05',end_date='2017-10-22')
val_df = atecml.data.filter_date(data,start_date='2017-10-23',end_date='2018-10-15')

predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_SUM]

feature_tree_num = 20
if (os.path.exists('./res_feature_list.dat')):
    print('Load Feature List from persistant store...')
    feature_list = joblib.load('./res_feature_list.dat')
else:
    print('Generate Random Feature List...')
    feature_list = {}
    for idx in range(0,feature_tree_num):
        feature_set = set(random.sample(predictors,120))
        feature_list[idx] = list(feature_set)
    joblib.dump(feature_list,'./res_feature_list.dat')

train_model =[]
for idx in range(0,1):
    for item in ['dart','gbdt','rf']:
        for feature_grp_idx in range(0,feature_tree_num):
            for target in ['Normal','Fraud']:
                train_id = item + '__'+target +'__'+str(idx) +'__' + str(feature_grp_idx) + '__res' 
                train_model.append(train_id)

trained_model_list =[]
with atecml.data.timer('Classification: Model Training'):
    for train_id in tqdm(range(len(train_model))):
        fit_model = model_train(train_df,val_df,feature_list,train_model[train_id])
        trained_model_list.append(fit_model)

