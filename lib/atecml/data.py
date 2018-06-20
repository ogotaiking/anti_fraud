import warnings
warnings.filterwarnings('ignore')


import copy as copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc


system_root_path = '/opt/ml/anti_ml/'
data_path = system_root_path + 'dataset/'

TRAIN_DATA = data_path + 'atec_anti_fraud_train.csv'
TEST_DATA = data_path + 'atec_anti_fraud_test_a.csv'

def load_train_data():
    train_df = pd.read_csv(TRAIN_DATA)
    train_df = train_df.sort_values(by='date').reset_index(drop='true')
    train_df['date'] = pd.to_datetime(train_df['date'],format="%Y%m%d")
    return train_df

def load_test_data():
    test_df = pd.read_csv(TEST_DATA)
    test_df['date'] = pd.to_datetime(test_df['date'],format="%Y%m%d")
    return test_df

def load_data():
    train_df = load_train_data()
    test_df = load_test_data()
    return (train_df,test_df)

#特征分析
def feature_describe(data_set):
    """
    feature_describe(train_df)
    """
    data_set_describe = data_set.describe()
    feature_describe = data_set_describe.T.reset_index().sort_values(by=['count','index'])
    return feature_describe

def feature_columns(data_set,LABEL_COL='label'):
    """
    根据dataset过滤Label，ID和Date，产生名特征列表
    """
    feature = list(data_set.columns);
    feature.remove(LABEL_COL)
    feature.remove('id')
    feature.remove('date')
    return feature

def train_df_filter_by_feature_cols(data_set,select_feature_col,request_date=False,request_id=True):
    """
    根据所选择的特征，补全id和label，并可选择是否补全时间列
    train_df_filter_by_feature_cols(train_df,['f1','f2'],True)
    """
    filter_list = copy.deepcopy(select_feature_col)
    if (request_id == True):
        filter_list.append('id')
    if (request_date == True):
        filter_list.append('date')
    filter_list.append('label')
    return data_set[filter_list]

def test_df_filter_by_feature_cols(data_set,select_feature_col,request_date=False):
    """
    根据所选择的特征，补全id，并可选择是否补全时间列
    用于特定特征训练时，预处理构造测试集和预测集数据
    test_df_filter_by_feature_cols(test_df,['f1','f2'],True)
    """
    filter_list = copy.deepcopy(select_feature_col)
    filter_list.append('id')
    if (request_date == True):
        filter_list.append('date')
    return data_set[filter_list]

def select_feature_by_count(feature_describe_df,counter):
    select_feature = list(feature_describe_df[feature_describe_df['count']==counter]['index'])
    if ('id'  in select_feature):
        select_feature.remove('id')
    if ('date' in select_feature):
        select_feature.remove('label')        
    if ('label' in select_feature):
        select_feature.remove('label')
    return select_feature

def feature_group_by_count(feature_describe):
    """
    根据特征描述（Feature Describe）矩阵，数据类型的数量对特征分组
    Example:
        fd = feature_describe(train_df)
        count_set,feature_group = feature_grp_by_count(fd)
        feature_group[count_set[1]]
    """
    feature_grp = {}
    count_list =[]
    for count_id in sorted(set(feature_describe['count'])):
        count_list.append(count_id)
        feature_list = select_feature_by_count(feature_describe,count_id)
        if ('id' in feature_list):
            feature_list.remove('id')
        if ('date' in feature_list):
            feature_list.remove('date')
        if ('label' in feature_list):
            feature_list.remove('label')        
        feature_grp[count_id] = feature_list
    return count_list[::-1],feature_grp

def data_label_split(data_set,LABEL_COL='label'):
    """
    用于将一个训练集标签和特征分离成两个DataFrame
    x,y= data_label_split(train_df)
    """
    data_set = data_set.reset_index(drop=True)
    X_data = data_set.drop([LABEL_COL],axis=1)
    Y_label = data_set[LABEL_COL]
    return (X_data,Y_label)   

def data_label_merge(X,Y):
    """
    用于将一个训练集标签和特征分离的两个DataFrame
    重新合成为一个集合， 
    使用时请注意顺序,最好做之间将他们设置好ID-INDEX
    
    df= data_label_split(X_train,Y_train)
    """    
    data_set =  pd.concat([X,Y],axis=1)
    data_set.reset_index(drop=True,inplace=True)
    return data_set

def train_test_split_by_date(data_set,split_date):
    """
    由于不能使用未来数据， 因此补充一个将训练集根据时间拆分的函数
    T_train,T_test= train_test_split_by_date(train_df,'2017-09-06')
    """
    Data_Before = data_set[data_set['date'] <= split_date]
    Data_After = data_set[data_set['date'] > split_date]
    return (Data_Before,Data_After)

def train_split_by_label(data_set):
    """
    将训练集根据标签拆分
    """
    Unknown = data_set[data_set['label']==-1]
    Normal = data_set[data_set['label']==0]
    Fraud =  data_set[data_set['label']==1]
    return (Unknown,Normal,Fraud)

def data_train_test_split(df,train_size=0.8):
    """
    随机划分测试集和训练集
    TT,Tt = data_split(train_df,0.3)
    """
    Train_Set = df.sample(frac=train_size)
    Test_Set = df.loc[~df.index.isin(Train_Set.index)]
    Train_Set  = shuffle(Train_Set)
    Test_Set  = shuffle(Test_Set)
    return (Train_Set,Test_Set)

def train_data_split_by_select_label(data_set,train_size=0.8,label_list=[0,1]):
    """
    为防止整体采样导致的有偏观测，对测试集进行分类标签的采样
    """
    Train_list = []
    Test_list = []
    for item in label_list:
        temp_df = data_set[data_set.label == item]
        temp_train,temp_test = data_train_test_split(temp_df,train_size)
        Train_list.append(temp_train)
        Test_list.append(temp_test)
    Train_DF= pd.concat(Train_list,axis=0)
    Test_DF = pd.concat(Test_list,axis=0)
    Train_DF  = shuffle(Train_DF)
    Test_DF  = shuffle(Test_DF)
    return Train_DF,Test_DF
    
def fillna_by_median(data_set):
    """
    基于中位数的NaN值填充
    """
    select_col = list(data_set.columns)
    for col_item in select_col:
        if ((col_item!='id') & (col_item!='label') & (col_item!='date')):
            data_set[col_item]=data_set[col_item].fillna(data_set[col_item].median())
    return data_set

def performance(y_test,y_predict_proba):
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


def Pre_Processing_data_for_Sklearn(data_set,feature_cols,select_label=[0,1],train_size=0.75,fillna="median",dropna=False):  
    """
    解决SKlearn中modelselect潜在的有偏估计
    
    X_train, X_test, y_train, y_test = Pre_Processing_data_for_Sklearn(train_df,['f1','f2'],[0,1],0.7)
    """
    filtered_data_set = train_df_filter_by_feature_cols(data_set,feature_cols,request_id=False)
    train_df,test_df = train_data_split_by_select_label(filtered_data_set,train_size=train_size,label_list=select_label)
    
    if (fillna=="median"):
        train_df= fillna_by_median(train_df)
        test_df= fillna_by_median(test_df)

    if (fillna!= False ):
        train_df= train_df.fillna(fillna)
        test_df= test_df.fillna(fillna)
        
    if (dropna == True):
        train_df= train_df.dropna()
        test_df= test_df.dropna()
        
    X_train,y_train = data_label_split(train_df)
    X_test,y_test = data_label_split(test_df)
    return (X_train, X_test, y_train, y_test)


def Pre_Processing_data_for_Sklearn_by_date(data_set,feature_cols,split_date,select_label=[0,1],fillna="median",dropna=False):  
    """
    根据时间来分割测试集和训练集
    
    X_train, X_test, y_train, y_test = Pre_Processing_data_for_Sklearn(train_df,['f1','f2'],'2017-10-11',[0,1])
    """
    train_df,test_df = train_test_split_by_date(data_set,split_date)    
    train_df = train_df_filter_by_feature_cols(train_df,feature_cols,request_id=False)
    test_df = train_df_filter_by_feature_cols(test_df,feature_cols,request_id=False)

    #build label_filter
    Train_list = []
    Test_list = []
    for item in select_label:
        temp_train = train_df[train_df.label == item]
        temp_test = test_df[test_df.label == item]
        Train_list.append(temp_train)
        Test_list.append(temp_test)
    Train_DF_after_label_filter = pd.concat(Train_list,axis=0)
    Test_DF_after_label_filter = pd.concat(Test_list,axis=0)
    #shuffle after concat
    Train_DF_after_label_filter = shuffle(Train_DF_after_label_filter)
    Test_DF_after_label_filter = shuffle(Test_DF_after_label_filter)
    
    if (fillna=="median"):
        Train_DF_after_label_filter = fillna_by_median(Train_DF_after_label_filter)
        Test_DF_after_label_filter  = fillna_by_median(Test_DF_after_label_filter)

    if (fillna!= False ):
        Train_DF_after_label_filter= Train_DF_after_label_filter.fillna(fillna)
        Test_DF_after_label_filter = Test_DF_after_label_filter.fillna(fillna)
        
    if (dropna == True):
        Train_DF_after_label_filter= Train_DF_after_label_filter.dropna()
        Test_DF_after_label_filter = Test_DF_after_label_filter.dropna()
        
    X_train,y_train = data_label_split(Train_DF_after_label_filter)
    X_test,y_test = data_label_split(Test_DF_after_label_filter)
    
    return (X_train, X_test, y_train, y_test)


def Pre_Processing_data_for_TF(X_train, X_test, y_train, y_test, predict_df, select_feature, LABEL='label'):
    """
    TODO：这个给TF准备的函数需要参考前述sklean的实现，照着改
    """
    FEATURES = copy.deepcopy(select_feature)
    training_set = pd.concat([X_train,y_train],axis=1)
    training_set.reset_index(drop=True,inplace=True)
    #label processing ?
    training_set['label'] = training_set['label'] +1
    
    test_set = pd.concat([X_test,y_test],axis=1)
    test_set.reset_index(drop=True,inplace=True)
    test_set['label'] = test_set['label'] +1
    
    #dropna or fillna ?
    prediction_set =predict_df[FEATURES].dropna()
    
    FEATURES.remove('date')
    FEATURES.remove('id')
    
    feature_cols =[tf.feature_column.numeric_column(k) for k in FEATURES]
    return (training_set,test_set,prediction_set,FEATURES,feature_cols)

def model_report(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['label'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    feature_importance = pd.DataFrame()
    feature_importance['feature_name'] = pd.Series(predictors)
    feature_importance['score'] = pd.Series(alg.feature_importances_)
    return feature_importance.sort_values(by='score',ascending=False)

