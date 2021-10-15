# coding: utf-8
######################## import 선언부 #######################
import re
import os
#import random
#import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import OneHotEncoder

#from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

################ 경로 설정 및 파일 read #######################
path = 'C:/Users/user/Dropbox/2021_2/intern/project_pm/'
load_path = path + 'pm_data/'  # 데이터 불러오는 경로
save_path = path + 'pm_data/prepro_taas/'  # 전처리 이후 데이터 저장하는 경로
result_path = path + 'pm_results/'
data = pd.read_csv(load_path + '어린이,노인,자전거도로,경사도포함/2017_2020_가해자자전거_PM사고정보병합(어린이,노인,자전거,경사도적용_OneHotEncoding_ver)_real.csv', encoding="EUC-KR")
data_m = pd.read_csv(load_path + 'prepro_taas/df_v2.csv', encoding="EUC-KR")
data_b = pd.read_csv(load_path + 'bus_tree/bus_n,tree_n추가.csv', encoding="EUC-KR")
data = data.iloc[:,1:]

################ 파일 합치기 #######################
n = 10
while n > 1 :
    n = len(data['Day']) - len(data_m['day'])
    string = '-1 '
    repeated_string = string*n
    splitted_string = repeated_string.split()
    temp = pd.concat([data_m['day'], pd.Series([int(i) for i  in splitted_string])], axis=0).reset_index(drop=True)
    data = data.drop(temp[temp != data['Day']].index.min()).reset_index(drop=True)
len(data), len(data_m), (data['Day'] == data_m['day']).sum()
(data_b['day'] == data_m['day']).sum()
data = data.drop(['at_age','vt_age'], axis = 1)

data.drop(list(data.filter(regex = 'at_[0-9]')), axis = 1, inplace = True)
data.drop(list(data.filter(regex = 'vt_[0-9]')), axis = 1, inplace = True)
data.rename(columns={'unknown':'at_prot_unknown'}, inplace=True)
data_m.rename(columns={'sidewalks':'road_num', 'sidewalks_y':'road_y'}, inplace=True)

df = pd.concat([data, data_m[['road_num','road_y','log_epdo','log_ari']]], axis=1)
df = pd.concat([df, data_m[[col for col in data_m.filter(regex = '.t_[0-9]').columns]]], axis = 1)
df = pd.concat([df, data_b[['busstop_n','tree_n']]], axis = 1)
df= df.rename(columns=str.lower)

df.describe()
df.isnull().sum()

df.loc[:,[(df[col].dtypes == 'object') for col in df.columns]]
df['acci_case_b'].value_counts()
df['tree_n'].value_counts()

##################### 데이터 가공 ##########################
df.drop(['year','month','day','hour','week','night','accident',
'd_acci','s_acci','c_acci','i_acci','road_form_b','at_gender','at_acci',
'vt_gender','vt_acci','alch','alch_cont','road_linear','road_straight'], axis=1, inplace=True)
df.drop(list(df.filter(regex = '.t_[0-9]')), axis = 1, inplace = True)

dt = df.copy()
dt_vv = dt[dt['acci_case_b']=='차대차']
dt_vp = dt[dt['acci_case_b']=='차대사람']
dt_va = dt[dt['acci_case_b']=='차량단독']
dt_vv = dt_vv.drop(labels='acci_case_b',axis=1)
dt_vp = dt_vp.drop(labels='acci_case_b',axis=1)
dt_va = dt_va.drop(labels='acci_case_b',axis=1)

dt_vv.head()
dt_vv = dt_vv.drop(labels=['epdo','log_epdo'],axis=1)
dt_vp = dt_vp.drop(labels=['epdo','log_epdo'],axis=1)
dt_va = dt_va.drop(labels=['epdo','log_epdo'],axis=1)
pd.value_counts(dt_vv['pm']).plot.bar()
plt.title('VV pm class histogram')
plt.xlabel('pm')
plt.ylabel('Frequency')
dt_vv['pm'].value_counts()
pd.value_counts(dt_vp['pm']).plot.bar()
plt.title('VP pm class histogram')
plt.xlabel('pm')
plt.ylabel('Frequency')
dt_vp['pm'].value_counts()
pd.value_counts(dt_va['pm']).plot.bar()
plt.title('VA pm class histogram')
plt.xlabel('pm')
plt.ylabel('Frequency')
dt_va['pm'].value_counts()
dt_vv_org = dt_vv.copy()
dt_vv.describe()
dt_vv.drop(['law_pedestrian','log_ari','busstop_n'], axis=1, inplace=True)
dt_vv.columns

##################### 모델링 ##########################
from pycaret.classification import *
data = dt_vv.sample(frac=0.95, random_state=786)
data_unseen = dt_vv.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

grid = setup(data=data, target='pm', fix_imbalance=True)
best_model = compare_models()
tuned_lda = tune_model(lda)
plot_model(tuned_lda, plot='learning')
plot_model(tuned_lda, plot='auc')
plot_model(tuned_lda, plot='confusion_matrix')
plot_model(tuned_lda, plot='feature')
evaluate_model(tuned_lda)

### 상호작용항
#운행 관련 변수 at_driving_b, law_cross, law_traffic, law_distance
#환경 관련 변수 road_flat, cross, cross_cnt, road_num, road_y, slope들 상호작용항 만들기
dt_vv2 = dt_vv.copy()

col_driving = np.array(['at_driving_b','law_cross','law_traffic','law_distance'])
col_env = np.array(['road_flat','cross','cross_cnt','road_num','road_y'])
col_slope = dt_vv2.columns[dt_vv2.columns.str.startswith('slope')]

for col1 in col_driving:
    for col2 in col_env:
        dt_vv2[col2 + '*' + col1] = dt_vv2[col1].mul(dt_vv2[col2])

for col1 in col_driving:
    for col2 in col_slope:
        dt_vv2[col2 + '*' + col1] = dt_vv2[col1].mul(dt_vv2[col2])

data = dt_vv2.sample(frac=0.95, random_state=786)
data_unseen = dt_vv2.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

cate = list(dt_vv2.columns)
cate.remove('cross_cnt')
cate.remove('tree_n')
cate.remove('road_num')
cate.remove('byca')
cate.remove('scholla')
cate.remove('silvera')
cate.remove('pm')

grid = setup(data=data, target='pm', fix_imbalance=True, categorical_features=cate, feature_selection=True)
best_model = compare_models()
lr = create_model('lr')
tuned_lr = tune_model(lr)

plot_model(tuned_lr, plot='learning')
plot_model(tuned_lr, plot='auc')
plot_model(tuned_lr, plot='confusion_matrix')
plot_model(tuned_lr, plot='feature')
evaluate_model(tuned_lr)
