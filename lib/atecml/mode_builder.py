import numpy as np
import pandas as pd
import tensorflow as tf
import data as data
from sklearn.externals import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor

train_df = pd.read_pickle('../data/train_df_with_nan_label.dat')

def model_fit(train_df,alg,alg_name,feature_list,feature_key,nan_string='',fill_mode="median",drop_mode=False,label_mode=0,start_date='2017-09-05',end_date='2017-09-06'):
    """
    #Label_Mode: 0-Normal,Fraud ; 1-Unknown,Normal
    """
    modelname = './sub_model/'+ alg_name + '___'+ str(feature_key) + '___' + str(fill_mode) + '___' + str(drop_mode) + '___' + str(label_mode)
    model_save_filename = modelname +'.model'
    model_parameters_name = modelname + '.dat'

    print("Start calculation ",modelname,".............")

    if (nan_string != ''):
        feature_list = nan_string.split('_')

    
    predictors = [x for x in train_df.columns if x not in feature_list+ ['id','date','label','nan_list']]
    filter_list = predictors + ['id','date','label','nan_list']

    model_parameters = dict()
    model_parameters['nan_list'] = nan_string
    model_parameters['fill_mode'] = fill_mode
    model_parameters['drop_mode'] = drop_mode
    model_parameters['start_date'] = start_date
    model_parameters['end_date'] = end_date
    model_parameters['label_mode'] = label_mode
    model_parameters['predictors'] = '___'.join(predictors)
    parameter_df = pd.DataFrame.from_dict(model_parameters,orient='index')
    parameter_df.to_pickle(model_parameters_name)

    train_feature_filter_df  = train_df[train_df['nan_list']== nan_string]
    train_feature_filter_df = train_feature_filter_df[filter_list]
    

    if (label_mode == 0 ):
        label_list=[0,1]
        train_df_filter = train_feature_filter_df[train_feature_filter_df['label'].isin(label_list)]
        train_df_filter['label'] = train_df_filter['label']

    #0-Unknown, 1-Normal
    if (label_mode == 1 ):
        label_list=[-1,0]
        train_df_filter = train_feature_filter_df[train_feature_filter_df['label'].isin(label_list)]
        train_df_filter['label'] = train_df_filter['label'] +1

    
    #Train_Test_Split_by_Date
    model_train = train_df_filter[ (train_df_filter['date'] >= start_date) &  (train_df_filter['date'] <= end_date)]

    #Fill&DROP
    if (fill_mode=="median"):
        for col_item in predictors:
            model_train[col_item]=model_train[col_item].fillna(model_train[col_item].median())

    if (fill_mode!= False ):
        for col_item in predictors:
            model_train[col_item]=model_train[col_item].fillna(fill_mode)

    if (drop_mode == True):
        model_train= model_train.dropna()

    target='label'
    if (len(model_train) >0):
        alg.fit(model_train[predictors],model_train[target])
        joblib.dump(alg, model_save_filename)


feature_group = list(train_df.groupby(['nan_list']).count().index)
feature_group.sort()


gbdt = GradientBoostingClassifier(max_depth=50,n_estimators=2000)
xgbc = XGBClassifier(max_depth=50,n_estimators=2000)
lr = LogisticRegression()
alg_name_list = ['GBDT','XGBOOST','LR']
alg_list = [gbdt,xgbc,lr]

key_list = []
for algo_idx in range(0,len(alg_list)):
    #for feature_idx in range(0,2):
    for feature_idx in range(0,len(feature_group)):
        for label_mode in range(0,2):
            key = str(algo_idx)+'__'+str(feature_idx) + '__' +str(label_mode)
            key_list.append(key)

def mt_calc(key):
    temp = key.split('__')
    algo_idx = int(temp[0])
    feature_idx = int(temp[1])
    label_mode = int(temp[2])
    alg = alg_list[algo_idx]
    alg_name = alg_name_list[algo_idx]
    nan_string = feature_group[feature_idx]
    feature_key = 'nan_list__' + temp[1]

    model_fit(train_df,alg,alg_name,'',feature_key,nan_string,False,True,label_mode,start_date='2017-09-05',end_date='2017-10-03')
    result = alg_name +'__'+ feature_key + '__label_mode=' + str(label_mode)
    return result 

with ProcessPoolExecutor() as pool:
    for result in pool.map(mt_calc,key_list):
        print(result+'...............fit done!')



