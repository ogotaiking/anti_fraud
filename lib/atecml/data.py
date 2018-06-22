import copy as copy
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from contextlib import contextmanager
from tqdm import tqdm
from time import strftime,time
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings('ignore')


system_root_path = '/opt/ml/anti_ml/'
data_path = system_root_path + 'dataset/'

TRAIN_DATA = data_path + 'atec_anti_fraud_train.csv'
TRAIN_DATA_DF_PICKLE = data_path + 'atec_anti_fraud_train.dat'

TEST_DATA = data_path + 'atec_anti_fraud_test_a.csv'
TEST_DATA_DF_PICKLE = data_path + 'atec_anti_fraud_test_a.dat'

NOT_FEATURE_COLUMNS = ['id','date','label','Normal','Fraud','NaN_LIST']


@contextmanager
def timer(func_name: str):
    """Elapsed Time
    """
    start = time()
    print('[{}][{}] Begin ...'.format(strftime('%Y-%m-%d %H:%M:%S'), func_name))
    yield
    print('[{}][{}] End   ...[Elapsed: {:.2f}s]'.format(strftime('%Y-%m-%d %H:%M:%S'), func_name, time()-start))
    

def load_train():
    #read from pickle cache to avoid csv parse and data preprocessing.
    if (os.path.exists(TRAIN_DATA_DF_PICKLE)):
        df = pd.read_pickle(TRAIN_DATA_DF_PICKLE)
    else:    
        df = pd.read_csv(TRAIN_DATA)
        df = df.sort_values(by='date').reset_index(drop='true')
        df['date'] = pd.to_datetime(df['date'],format="%Y%m%d")
        #group data based on Missing Value-List
        df['NaN_LIST'] = df.index.map(lambda x: '_'.join(list((df.iloc[x][df.iloc[x].isnull() == True]).index)))
        
        # Create a new feature for normal & fraud transaction.
        # Then training target could be based on 'Normal' or 'Fraud'
        # This new cols filter the unknown class..
        df.loc[df.label == 0, 'Normal'] = 1
        df.loc[df.label != 0, 'Normal'] = 0
        df.loc[df.label == 1, 'Fraud'] = 1
        df.loc[df.label != 1, 'Fraud'] = 0        
        df.to_pickle(TRAIN_DATA_DF_PICKLE)        
    return df

def load_test():
    if (os.path.exists(TEST_DATA_DF_PICKLE)):
        df = pd.read_pickle(TEST_DATA_DF_PICKLE)
    else:    
        df = pd.read_csv(TEST_DATA)
        df['date'] = pd.to_datetime(df['date'],format="%Y%m%d")
        df['NaN_LIST'] = df.index.map(lambda x: '_'.join(list((df.iloc[x][df.iloc[x].isnull() == True]).index)))
        df.to_pickle(TEST_DATA_DF_PICKLE)
    return df

def load():
    train_df = load_train()
    test_df = load_test()
    return (train_df,test_df)

def filter_date(df,start_date='2001-01-01',end_date='2099-01-01'):
    return df[(df['date'] >= start_date) &(df['date'] <= end_date)]

def feature_columns(df):
    result_list = [x for x in df.columns if x not in NOT_FEATURE_COLUMNS]
    return result_list

def fillna_by_DateMedian(df,target=-2):
    date_list = list(set(df['date']))
    tdf_list = []
    predictors = [x for x in df.columns if x not in NOT_FEATURE_COLUMNS]
    with timer('Fillna by DateMedian'):
        for idx in tqdm(range(len(date_list))):
            date = date_list[idx]
            tdf = df[df['date'] == date]
            if(target == -2):
                values = tdf[predictors].median().to_dict()
            else:
                values = tdf[tdf['label']==target][predictors].median().to_dict()
            tdfn = tdf.fillna(value=values)
            tdf_list.append(tdfn)
        result = pd.concat(tdf_list)
    return result.sort_index()

    
def fillna_by_median(data_set):
    """
    基于中位数的NaN值填充
    """
    select_col = list(data_set.columns)
    for col_item in select_col:
        if ((col_item!='id') & (col_item!='label') & (col_item!='date')):
            data_set[col_item]=data_set[col_item].fillna(data_set[col_item].median())
    return data_set



def accuracy_validation(y_test,y_predict_proba):
    """
    基于ROC的模型性能测量，并根据蚂蚁金服评分标准输出分数
    """
    fpr, tpr, thresholds = roc_curve(y_test,y_predict_proba)
    roc_auc = auc(fpr, tpr)
    roc_result = pd.DataFrame()
    roc_result['fpr'] = pd.Series(fpr)
    roc_result['tpr'] = pd.Series(tpr)
    roc_result['thresholds'] = pd.Series(thresholds)
    TPR1= float(roc_result[roc_result['fpr']<=0.001002].tail(1)['tpr'])
    TPR2=float(roc_result[roc_result['fpr']<=0.005002].tail(1)['tpr'])
    TPR3=float(roc_result[roc_result['fpr']<=0.010002].tail(1)['tpr'])
    FINAL_SCORE = 0.4*TPR1 + 0.3*TPR2 + 0.3 * TPR3
    print(FINAL_SCORE)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return (FINAL_SCORE,roc_result,roc_auc)
