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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

model = {}

gbdt_params = {'boosting_type': 'gbdt',
               'n_estimators': 200,
               'use_missing' : True,
               'categorical_feature': 0,
               'is_unbalance': True,               
               'max_depth': -1,
               'num_leaves': 80,
               'learning_rate': 0.05, 
               'max_bin': 512, 
               'subsample_for_bin': 200,
               'subsample': 0.8, 
               'subsample_freq': 1, 
               'colsample_bytree': 0.8, 
               'reg_alpha': 5, 
               'reg_lambda': 10,
               'min_split_gain': 0.5, 
               'min_child_weight': 1, 
               'min_child_samples': 5, 
               'scale_pos_weight': 1,
               'num_class' : 1,
               'metric' : 'binary_error',
               'seed': 42,
               'nthread': -1
              }


model["GBDT"] = LGBMClassifier(**gbdt_params)
#model["GOSS"] = LGBMClassifier(**goss_params)
#model["DART"] = LGBMClassifier(**dart_params)

#Loading Data....
train_df,test_df = atecml.data.load()
predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_COLUMNS]


imp_feature = [ 'f105', 'f106', 'f14', 'f15', 'f17', 'f18', 'f185', 'f19', 'f204', 'f208', 'f209', 'f21', 'f210', 'f215', 'f217', 'f218', 'f23', 'f234', 'f235', 'f236', 'f237', 'f238', 'f241', 'f242', 'f243', 'f244', 'f245', 'f247', 'f248', 'f253', 'f262', 'f27', 'f278', 'f31', 'f33', 'f57', 'f6', 'f7', 'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f91']

feature_tree_num = 50
if (os.path.exists('./feature_list.dat')):
    print('Load Feature List from persistant store...')
    feature_list = joblib.load('./feature_list.dat')
else:
    print('Generate Random Feature List...')
    feature_list = {}
    predictors_wo_f5 = copy.deepcopy(predictors)
    predictors_wo_f5.remove('f5')
    for idx in range(0,feature_tree_num):
        feature_set = set(imp_feature + random.sample(predictors_wo_f5,20))
        feature_list[idx] = ['f5'] + list(feature_set)
    joblib.dump(feature_list,'./feature_list.dat')


train_df = atecml.data.filter_date(train_df,start_date='2017-09-05',end_date='2017-10-15')


def model_train(df, feature_list,model_name):
    model_cache_name = './'+model_name+'.model'
    if (os.path.exists(model_cache_name)):
        clf = joblib.load(model_cache_name)
    else:
        params = model_name.split('__')
        model_key = params[0]
        target = params[1]
        clf = model[model_key]
        f_idx = int(params[3])
        select_feature = feature_list[f_idx]
        with atecml.data.timer('> {} <: Training...'.format(model_name)):
            clf.fit(df[select_feature],df[target])
        joblib.dump(clf,model_cache_name)
    return clf

train_model =[]
for idx in range(0,1):
    for item in model.keys():
        for feature_grp_idx in range(0,feature_tree_num):
            for target in ['Fraud']:
                train_id = item + '__'+target +'__'+str(idx) +'__' + str(feature_grp_idx)
                train_model.append(train_id)

trained_model_list =[]
with atecml.data.timer('Classification: Model Training'):
    for train_id in tqdm(range(len(train_model))):
        fit_model = model_train(train_df,feature_list,train_model[train_id])
        trained_model_list.append(fit_model)

