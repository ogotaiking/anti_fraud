import copy as copy
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from contextlib import contextmanager
from tqdm import tqdm
from time import strftime,time
from sklearn.metrics import roc_curve, auc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')


system_root_path = '/opt/ml/anti_ml/'
data_path = system_root_path + 'dataset/'

TRAIN_DATA = data_path + 'atec_anti_fraud_train.csv'
TRAIN_DATA_DF_PICKLE = data_path + 'atec_anti_fraud_train.dat'

TEST_DATA = data_path + 'atec_anti_fraud_test_b.csv'
TEST_DATA_DF_PICKLE = data_path + 'atec_anti_fraud_test_b.dat'

WRONG_DISTRIBUTION = ['f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f111', 'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125', 'f126', 'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f135', 'f136', 'f137', 'f138', 'f139', 'f140', 'f141', 'f142', 'f143', 'f144', 'f145', 'f146', 'f147', 'f148', 'f149', 'f150', 'f151', 'f152', 'f153']

NOT_FEATURE_COLUMNS = ['id','date','label','Normal','Fraud','NaN_LIST']

NOT_FEATURE_COLUMNS2 = WRONG_DISTRIBUTION +NOT_FEATURE_COLUMNS

HIGH_MISSING_RATE_FEATURE = ['f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47','n36', 'n37', 'n38', 'n39', 'n40', 'n41', 'n42', 'n43', 'n44', 'n45', 'n46', 'n47']

NOT_IMP_FEATURE =['f114', 'f231', 'f280', 'f154', 'f251', 'f208', 'f127', 'f292', 'f157', 'f223', 'f189', 'f229', 'f255', 'f51', 'f230', 'f253', 'f27', 'f143', 'f39', 'f120', 'f209', 'f67', 'f146', 'f142', 'f59', 'f121', 'f22', 'f151', 'f248', 'f293', 'f125', 'f232', 'f165', 'f200', 'f290', 'f40', 'f225', 'f35', 'f136', 'f69', 'f138', 'f135', 'f26', 'f49', 'f163', 'f145', 'f227', 'f139', 'f252', 'f118', 'f164', 'f113', 'f116', 'f119', 'f222', 'f242', 'f297', 'f206', 'f152', 'f47', 'f243', 'f218', 'f46', 'f207', 'f131', 'f214', 'f140', 'f115', 'f130', 'f228', 'f213', 'f45', 'f294', 'f37', 'f250', 'f238', 'f70', 'f249', 'f272', 'f126', 'f281', 'f124', 'f42', 'f128', 'f122', 'f38', 'f148', 'f166', 'f137', 'f147', 'f123', 'f133', 'f60', 'f149', 'f217', 'f112', 'f268', 'f144', 'f287', 'f220', 'f36', 'f109', 'f44', 'f221', 'f50', 'f216', 'f226', 'f153', 'f48', 'f111', 'f54', 'f129', 'f23', 'f75', 'f233', 'f43', 'f41', 'f224', 'f247', 'f74', 'f63', 'f117', 'f196', 'f289', 'f211', 'f246', 'f141', 'f295', 'f286', 'f288', 'f71', 'f212', 'f132', 'f150']

NOT_FEATURE_SUM = NOT_FEATURE_COLUMNS + NOT_IMP_FEATURE

CATE_FEATURE_LIST = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f28', 'f29', 'f61', 'f87', 'f88', 'f89', 'f90', 'f98', 'f99', 'f155', 'f156', 'f158', 'f159', 'f160', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f179', 'f180', 'f181', 'f182', 'f183', 'f186', 'f187', 'f188', 'f190', 'f191', 'f194', 'f195', 'f197', 'f198', 'f199', 'f201', 'f202', 'f203', 'f254', 'f256', 'f257', 'f258', 'f267', 'f269', 'f273', 'f274', 'f275', 'f276', 'f277']

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
    #data_set.fillna(data_set.mean(axis=0))
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
    print('Ant-Score:',FINAL_SCORE)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return (FINAL_SCORE,roc_result,roc_auc)

def calc_iv(df,feature,target='label',category_num = 128, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """
    with timer('WoE/IV Calculation for '+feature):
        lst = []

        n_unique = df[feature].nunique()
        if ((n_unique <= category_num) or (feature == 'f5')):
            feature_type = 'category'
            df[feature] = df[feature].fillna("NULL")
            _f_cat = list(df[feature].unique())
            for i in range(len(_f_cat)):
                val = _f_cat[i]
                lst.append([feature,                                                        # Variable
                            val,                                                            # Value
                            df[df[feature] == val].count()[feature],                        # All
                            df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                            df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)
        else:
            feature_type = 'value'
            low_band = 0
            high_band = df[df[target]==1][feature].max()            
            delta = (high_band - low_band) / category_num
            df[feature] = df[feature].fillna(-1)
            val = -1
            lst.append([feature,                                                       
                        val,                                                            
                        df[df[feature] == val].count()[feature],                        
                        df[(df[feature] == val) & (df[target] == 0)].count()[feature],  
                        df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) 
            
            for idx in range(0,category_num):
                _low = low_band + idx * delta
                _high = _low + delta
                val = _low
                _filter = (df[feature] > _low) & (df[feature] <= _high)
                lst.append([feature,                                                        # Variable
                            val,                                                            # Value
                            df[_filter].count()[feature],                        # All
                            df[ _filter & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                            df[ _filter & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)                
            
            val = high_band
            lst.append([feature,                                                       
                        val,                                                            
                        df[df[feature] > val].count()[feature],                        
                        df[(df[feature] > val) & (df[target] == 0)].count()[feature],  
                        df[(df[feature] > val) & (df[target] == 1)].count()[feature]])    
            
    
        data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    
        data['Share'] = data['All'] / data['All'].sum()
        data['Bad Rate'] = data['Bad'] / data['All']
        
        data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
        data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
        data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    
        data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        data['WoE'] = data['WoE'].fillna(0)
    
        data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    
        data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
        data.index = range(len(data.index))
    
        if pr:
            print(data)
            print('IV = ', data['IV'].sum())
    
    
        iv = data['IV'].sum()
        # print(iv)
        
        iv_dict = {}
        iv_dict[feature] = iv
        woe_dict ={}
        woe_dict[feature] = data
        type_dict ={}
        type_dict[feature] = feature_type

    return iv_dict,woe_dict,type_dict


def woe_convert(df,feature,woe_table,type_list,category_num = 128):
    if (type_list[feature] == 'category'):
        woe_dict = {}
        for idx in range(len(woe_table[feature])):
            key = woe_table[feature].loc[idx]['Value']
            woe = woe_table[feature].loc[idx]['WoE']
            woe_dict[key] = woe
        fff = df[feature].fillna('NULL')
        result = fff.map(lambda x : woe_dict[x] if x in woe_dict.keys() else 0)
    else:
        woe_list = list(woe_table[feature]['WoE'])
        _l2,_l1 = list(woe_table[feature]['Value'].tail(2))
        delta = _l1 - _l2
        fff = (np.floor(df[feature] / delta)+1).fillna(0).map(lambda x : x if x <= category_num else category_num+1)
        result = fff.map(lambda x : woe_list[int(x)])
    return result
