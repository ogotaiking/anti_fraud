import os

import numpy as np
import pandas as pd
import tensorflow as tf

import atecml.data

from contextlib import contextmanager
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

model = {}

#model["RandomForest"] = RandomForestClassifier(n_estimators=1000, max_depth=50, n_jobs=-1)
#model["ExtraTree"] =ExtraTreesClassifier(n_estimators=1000, max_depth=50, n_jobs=-1)
model["LightGBM"] = LGBMClassifier(n_estimators=1000, max_depth=50)
#model["GBDT"] =GradientBoostingClassifier(n_estimators=1000, max_depth=50)
model["XGBOOST"] =XGBClassifier(n_estimators=1000, max_depth=50,nthread=80)

#Loading Data....
train_df,test_df = atecml.data.load()
predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_COLUMNS]

train_df = atecml.data.filter_date(train_df,start_date='2017-09-05',end_date='2017-10-15')

with atecml.data.timer('PreProcessing: fillna'):
    for idx in tqdm(range(len(predictors))):
        item = predictors[idx]
        train_df[item].fillna(train_df[item].min(), inplace=True)

def model_train(df, predictors,model_name):
    model_cache_name = './'+model_name+'.model'
    if (os.path.exists(model_cache_name)):
        clf = joblib.load(model_cache_name)
    else:
        params = model_name.split('__')
        model_key = params[0]
        target = params[1]
        clf = model[model_key]
        with atecml.data.timer('> {} <: OverSample for imbalance data'.format(model_key)):
            X_resampled, y_resampled = SMOTE().fit_sample(df[predictors],df[target])
        with atecml.data.timer('> {} <: Training...'.format(model_key)):
            clf.fit(X_resampled,y_resampled)
        joblib.dump(clf,model_cache_name)
    return clf

train_model =[]
for idx in range(0,2):
    for item in model.keys():
        for target in ['Normal','Fraud']:
            train_id = item + '__'+target +'__'+str(idx) 
            train_model.append(train_id)


trained_model_list =[]
with atecml.data.timer('Classification: Model Training'):
    for train_id in tqdm(range(len(train_model))):
        fit_model = model_train(train_df,predictors,train_model[train_id])
        trained_model_list.append(fit_model)

