import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf

import atecml.data

from contextlib import contextmanager
from tqdm import tqdm
from time import strftime,time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@contextmanager
def timer(func_name: str):
    """Elapsed Time
    """
    start = time()
    print('[{}][{}] Begin ...'.format(strftime('%Y-%m-%d %H:%M:%S'), func_name))
    yield
    print('[{}][{}] End   ...[Elapsed: {:.2f}s]'.format(strftime('%Y-%m-%d %H:%M:%S'), func_name, time()-start))

train_df,test_df = atecml.data.load()
from sklearn.metrics import log_loss
from MLFeatureSelection import FeatureSelection as FS
import lightgbm as lgb
predictors = [x for x in train_df.columns if x not in atecml.data.NOT_FEATURE_COLUMNS]

with timer('PreProcessing: fillna'):
    for idx in tqdm(range(len(predictors))):
        item = predictors[idx]
        train_df[item].fillna(train_df[item].min(), inplace=True)

from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

def score(y_test,y_predict_proba):
    fpr, tpr, thresholds = roc_curve(y_test,y_predict_proba)
    #roc_auc = auc(fpr, tpr)
    score = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.01)[0][0]]
    return score

def validation(X, Y, features, clf, lossfunction):
    totaltest = []
    kf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X.ix[train_index,:][features], X.ix[test_index,:][features]
        y_train, y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, y_train)
        totaltest.append(lossfunction(y_test, clf.predict(X_test)))
    return np.mean(totaltest)

import lightgbm as lgb

sf = FS.Select(Sequence = True, Random = False, Cross = False)
sf.ImportDF(train_df, label ='Fraud') 
sf.ImportLossFunction(score, direction = 'ascend')
sf.InitialNonTrainableFeatures(atecml.data.NOT_FEATURE_COLUMNS) 
sf.InitialFeatures(predictors) 
sf.GenerateCol() 
sf.SetSample(1, samplemode = 1) 
sf.SetTimeLimit(720)
sf.clf = lgb.LGBMClassifier(random_state=10, num_leaves =50, n_estimators=1000, max_depth=10, learning_rate = 0.1, n_jobs=-1)
sf.SetLogFile('record.log') 
sf.run(validation) 
