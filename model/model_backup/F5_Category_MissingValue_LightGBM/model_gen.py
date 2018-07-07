import os

import numpy as np
import pandas as pd
import tensorflow as tf

import atecml.data

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
               'n_estimators': 600,
               'use_missing' : True,
               'categorical_feature': 4,
               'is_unbalance': True,               
               'max_depth': -1,
               'num_leaves': 64, 
               'learning_rate': 0.05, 
               'max_bin': 512, 
               'subsample_for_bin': 200,
               'subsample': 1, 
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
               'nthread': -1}


goss_params = {'boosting_type': 'goss',
               'n_estimators': 600,
               'use_missing' : True,
               'categorical_feature': 4,
               'is_unbalance': True,               
               'max_depth': -1,
               'num_leaves': 64, 
               'learning_rate': 0.05, 
               'max_bin': 512, 
               'subsample_for_bin': 200,
               'subsample': 1, 
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
               'nthread': -1}


dart_params = {'boosting_type': 'dart',
               'n_estimators': 600,
               'use_missing' : True,
               'categorical_feature': 4,
               'is_unbalance': True,               
               'max_depth': -1,
               'num_leaves': 64, 
               'learning_rate': 0.05, 
               'max_bin': 512, 
               'subsample_for_bin': 200,
               'subsample': 1, 
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
               'nthread': -1}

model["GBDT"] = LGBMClassifier(**gbdt_params)
model["GOSS"] = LGBMClassifier(**goss_params)
#model["DART"] = LGBMClassifier(**dart_params)

#Loading Data....
train_df,test_df = atecml.data.load()
predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_COLUMNS]

feature_list = {}
feature_list[0] = predictors
feature_list[1] = ['f7', 'f248', 'f238', 'f210', 'f5' , 'f218', 'f6', 'f215', 'f82', 'f247', 'f234', 'f244',  'f237', 'f245', 'f246', 'f18', 'f253', 'f243', 'f217', 'f236']
feature_list[2] = ['f7', 'f248', 'f238', 'f210', 'f5', 'f218', 'f6', 'f215', 'f82', 'f247', 'f234', 'f244',  'f237', 'f245', 'f246', 'f18', 'f253', 'f243', 'f217', 'f236', 'f222', 'f15', 'f106', 'f216', 'f17', 'f235', 'f86', 'f84', 'f85', 'f242', 'f19', 'f208', 'f14', 'f4', 'f209', 'f263', 'f207', 'f233', 'f101', 'f252', 'f204', 'f214', 'f83', 'f58', 'f57', 'f163', 'f231', 'f240', 'f53', 'f164']
feature_list[3] = ['f7', 'f248', 'f238', 'f210', 'f5','f218', 'f6', 'f215', 'f82', 'f247', 'f234', 'f244',  'f237', 'f245', 'f246', 'f18', 'f253', 'f243', 'f217', 'f236', 'f222', 'f15', 'f106', 'f216', 'f17', 'f235', 'f86', 'f84', 'f85', 'f242', 'f19', 'f208', 'f14', 'f4', 'f209', 'f263', 'f207', 'f233', 'f101', 'f252', 'f204', 'f214', 'f83', 'f58', 'f57', 'f163', 'f231', 'f240', 'f53', 'f164', 'f262', 'f226', 'f206', 'f221', 'f225', 'f11', 'f232', 'f251', 'f230', 'f25', 'f229', 'f162', 'f54', 'f52', 'f105', 'f205', 'f241', 'f223', 'f81', 'f161', 'f249', 'f49', 'f224', 'f63', 'f27', 'f34', 'f32', 'f21', 'f220', 'f80', 'f51', 'f8', 'f227', 'f12', 'f35', 'f48', 'f55', 'f239', 'f219', 'f213', 'f30', 'f100', 'f103', 'f211', 'f185', 'f50', 'f183', 'f28', 'f228', 'f278']


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

        X_train,X_test,y_train,y_test = train_test_split(df[select_feature],df[target],test_size=0.1, random_state=42)
        '''
        with atecml.data.timer('> {} <: OverSample for imbalance data'.format(model_key)):
            X_resampled, y_resampled = SMOTE().fit_sample(df[predictors],df[target])
        '''
        with atecml.data.timer('> {} <: Training...'.format(model_name)):
            clf.fit(X_train,y_train)
        joblib.dump(clf,model_cache_name)
    return clf

train_model =[]
for idx in range(0,10):
    for item in model.keys():
        for target in ['Normal','Fraud']:
            for feature_grp_idx in range(0,4):
                train_id = item + '__'+target +'__'+str(idx) +'__' + str(feature_grp_idx)
                train_model.append(train_id)


trained_model_list =[]
with atecml.data.timer('Classification: Model Training'):
    for train_id in tqdm(range(len(train_model))):
        fit_model = model_train(train_df,feature_list,train_model[train_id])
        trained_model_list.append(fit_model)




