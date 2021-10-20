# coding: utf-8
######################## import 선언부 #######################
import re
import os
import random
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')
############## datafarme 보여주는 범위 설정 ###################
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.2f}'.format
################ 경로 설정 및 파일 read #######################
path = 'C:/Users/user/Dropbox/2021_2/intern/project_pm/'
load_path = path + 'pm_data/'  # 데이터 불러오는 경로
save_path = path + 'pm_data/prepro_taas/'  # 전처리 이후 데이터 저장하는 경로
result_path = path + 'pm_results/'
data = pd.read_csv(load_path + 'taas/2017_2020_서울_가해자PM_자전거_이륜차사고정보통합.csv', encoding='EUC-KR')
cate = pd.read_csv(load_path + 'taas/catego2.csv', encoding='EUC-KR')
# 세가지 차종만
dt=data[(data['차량용도가해자']=='개인형이동수단(PM)')|(data['차량용도가해자']=='이륜차')|(data['차량용도가해자']=='자전거')]
# 삭제할 features drop하기
dt.drop([x for x in cate[cate['method']=='delete'].iloc[:,0]],axis=1, inplace=True)
df=dt
dt.isnull().sum()
cate['method'].unique()
dummy = [x for x in cate[cate['method']=='dummy'].iloc[:,0]]
binary = [x for x in cate[cate['method']=='binary'].iloc[:,0]]
ordinal = [x for x in cate[cate['method']=='ordinal'].iloc[:,0]]
leave = [x for x in cate[cate['method']=='LeaveOneOut'].iloc[:,0]]
lvor = [x for x in cate[cate['method']=='LeaveOneOut/ordinal'].iloc[:,0]]
binning = [x for x in cate[cate['method']=='binning'].iloc[:,0]]
numeric = [x for x in cate[cate['method']=='numeric'].iloc[:,0]]
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
def make_week(x):  # 요일 데이터로 변환
    week=['일','월','화','수','목','금','토']
    for i in range(0,7):
      if week[i]==x[0:1]:
        return i
    
def dummy_encoding(var, col): # encoding 패키지를 사용하기 때문에 df[[]] 대괄호 두개
    global df
    enc = OneHotEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(df[var]).toarray())
    for i in range(len(enc_df.columns)):
        enc_df.iloc[:,i] = enc_df.iloc[:,i]
    enc_df.columns = col
    enc_df.iloc[:,:-1]
    if len(df) == len(enc_df):
        df = pd.concat([df.reset_index(drop=True),enc_df.reset_index(drop=True)],axis=1)
        df.drop(var, axis=1, inplace=True)
    else:
        print('col len error')

def binary_encoding(var, atr, newname): # 변수, 1로 변환할 변수명, 새로운 변수 이름
    global df
    bn = []
    for i in range(len(df[var])):
        if df[var].iloc[i,0] == atr[0]:
            bn.append(1)
        elif df[var].iloc[i,0] == atr[1]:
            bn.append(0)
        else:
            bn.append(-1)
        
    bv = pd.DataFrame(bn)
    bv.columns = [newname]
    if len(df) == len(bv):
        df = pd.concat([df.reset_index(drop=True),bv.reset_index(drop=True)], axis = 1)
        df.drop(var, axis=1, inplace=True)
    else:
        print('col len error')
    
def ordinal_encoding(var, atr, newname): # 변수, 변수 속성 순서 지정, 새로운 변수이름
    global df
    odn = pd.Categorical(df[var], categories = atr, ordered = False) # atr 에서 순서 지정해줘야함
    odn_v = pd.DataFrame(odn.codes)
    odn_v.columns = [newname]
    if len(df) == len(odn_v):
        df = pd.concat([df.reset_index(drop=True),odn_v.reset_index(drop=True)], axis = 1)
        df.drop(var, axis=1, inplace=True)
    else:
        print('col len error')
# ## 예시 
# # 1. dummy
# df['week'] = df['요일'].apply(make_week)
# dummy_encoding(['week'], ["sun","mon","tue","wed","thu","fri","sat"])

# # 2. binary
# binary_encoding(['주야'], "야", "night")

# # 3. ordinal
# ordinal_encoding('사고내용', ['부상신고', '경상', '중상', '사망'], "accident")
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
#tagert encoding & leave-one-out encoding source : https://brendanhasz.github.io/2019/03/04/target-encoding
# Brendan Hasz, 2019

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.

    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target 
            encode all categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X 
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)

class TargetEncoderLOO(TargetEncoder):
    """Leave-one-out target encoder.
    """
    
    def __init__(self, cols=None):
        """Leave-one-out target encoding for categorical features.
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.
        """
        self.cols = cols
        

    def fit(self, X, y):
        """Fit leave-one-out target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to target encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.sum_count = dict()
        for col in self.cols:
            self.sum_count[col] = dict()
            uniques = X[col].unique()
            for unique in uniques:
                ix = X[col]==unique
                self.sum_count[col][unique] = \
                    (y[ix].sum(),ix.sum())
            
        # Return the fit object
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses leave-one-out target encoding for the training fold,
        and uses normal target encoding for the test fold.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Use normal target encoding if this is test data
        if y is None:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    vals[X[col]==cat] = sum_count[0]/sum_count[1]
                Xo[col] = vals

        # LOO target encode each column
        else:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col]==cat
                    vals[ix] = (sum_count[0]-y[ix])/(sum_count[1]-1)
                Xo[col] = vals
            
        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
def target_encoding(X,y):
    y=y.astype(str)
    enc=ce.OneHotEncoder().fit(y)
    y_onehot=enc.transform(y)
    class_names=y_onehot.columns
    
    X_new=pd.DataFrame()
    name=['pm','motor']
    for i in range(len(class_names[:-1])):
        cl = class_names[:-1][i]
        enc=TargetEncoder()
        temp=enc.fit_transform(X,y_onehot[cl])
        temp.columns=[str(x)+'_'+str(name[i]) for x in temp.columns]
        X_new=pd.concat([X_new,temp],axis=1)
    return X_new

def leave_encoding(X,y):
    y=y.astype(str)
    enc=ce.OneHotEncoder().fit(y)
    y_onehot=enc.transform(y)
    class_names=y_onehot.columns
    
    X_new=pd.DataFrame()
    name=['pm','motor']
    for i in range(len(class_names[:-1])):
        cl = class_names[:-1][i]
        enc=TargetEncoderLOO()
        temp=enc.fit_transform(X,y_onehot[cl])
        temp.columns=[str(x)+'_'+str(name[i]) for x in temp.columns]
        X_new=pd.concat([X_new,temp],axis=1)
    return X_new
# just in case, 3개짜리 남겨둠
#def leave_encode_3(X,y):
    #y=y.astype(str)
    #enc=ce.OneHotEncoder().fit(y)
    #y_onehot=enc.transform(y)
    #class_names=y_onehot.columns
    
    #X_new=pd.DataFrame()
    #name=['pm','motor','bi']
    #for i in range(len(class_names)):
        #cl = class_names[i]
        #enc=TargetEncoderLOO()
        #temp=enc.fit_transform(X,y_onehot[cl])
        #temp.columns=[str(x)+'_'+str(name[i]) for x in temp.columns]
        #X_new=pd.concat([X_new,temp],axis=1)
    #return X_new
# # 예시
# X_lvenc=leave_encoding(df[['사고유형_중분류','법규위반가해자']],df['차량용도가해자'])
# df = pd.concat([df,X_lvenc],axis=1)
# df.drop(['사고유형_중분류','법규위반가해자'],axis=1,inplace=True)
# def one_hot_encoding_cate(var):
#     enc = OneHotEncoder()
#     enc_df = pd.DataFrame(enc.fit_transform(var).toarray())
#     print(enc.categories_)

# one_hot_encoding_cate(df[[dummy[1]]])
# 1. dummy
df['week'] = df[dummy[0]].apply(make_week)

dummy_encoding(['week'], ["sun","mon","tue","wed","thu","fri","sat"])
dummy_encoding([dummy[1]], ["pm","motor","bi"])

df.drop('요일',axis=1,inplace=True)
binary
# 3. ordinal
ordinal_encoding(ordinal[0], ['부상신고', '경상', '중상', '사망'], "accident")
ordinal_encoding(ordinal[1], ['직진관련', '기타', '불명', '회전관련','주행 중 대기','주정차중','후진 중'], "at_driving_straight")
ordinal_encoding(ordinal[2], ['주차장', '단일로', '교차로','기타','불명'], "road_form")
df = df2
df2 = df.copy()
straight = []
for i in range(len(df['at_driving_straight'])):
    val = df['at_driving_straight'][i]
    if val == 0:
        straight.append(1.00)
    elif val in (1,2):
        straight.append(-1.00)
    else:
        straight.append(0.00)
repnum = round((pd.DataFrame(straight) != -1.00).mean()[0],4)
for i in range(len(straight)):
    val = straight[i]
    if val == -1:
        straight[i] = repnum
    else:
        straight[i] = val
leftright = []
for i in range(len(df['at_driving_straight'])):
    val = df['at_driving_straight'][i]
    if val == 3:
        leftright.append(1.00)
    elif val in (1,2):
        leftright.append(-1.00)
    else:
        leftright.append(0.00)
repnum2 = round((pd.DataFrame(leftright) != -1.00).mean()[0],4)
for i in range(len(leftright)):
    val = leftright[i]
    if val == -1:
        leftright[i] = repnum2
    else:
        leftright[i] = val
roadform = []
for i in range(len(df['road_form'])):
    val = df['road_form'][i]
    if val in (3,4):
        roadform.append(-1.00)
    else:
        straight.append(float(val))
df[leave]
X_lvenc=leave_encoding(df[leave],df['차량용도가해자'])
df = pd.concat([df,X_lvenc],axis=1)
df.drop(['사고유형_중분류','법규위반가해자'],axis=1,inplace=True)
df.drop(leave,axis=1,inplace=True)
df = df2
straight = []
for i in range(len(df['at_driving_straight'])):
    val = df['at_driving_straight'][i]
    if val == 0:
        straight.append(1.00)
    elif val in (1,2):
        straight.append(-1.00)
    else:
        straight.append(0.00)
repnum = round((pd.DataFrame(straight) != -1.00).mean()[0],4)
for i in range(len(straight)):
    val = straight[i]
    if val == -1:
        straight[i] = repnum
    else:
        straight[i] = val
leftright = []
for i in range(len(df['at_driving_straight'])):
    val = df['at_driving_straight'][i]
    if val == 3:
        leftright.append(1.00)
    elif val in (1,2):
        leftright.append(-1.00)
    else:
        leftright.append(0.00)
repnum2 = round((pd.DataFrame(leftright) != -1.00).mean()[0],4)
for i in range(len(leftright)):
    val = leftright[i]
    if val == -1:
        leftright[i] = repnum2
    else:
        leftright[i] = val
roadform = []
for i in range(len(df['road_form'])):
    val = df['road_form'][i]
    if val in (3,4):
        roadform.append(-1.00)
    else:
        straight.append(float(val))
X_lvenc=leave_encoding(df[leave],df['차량용도가해자'])
df = pd.concat([df,X_lvenc],axis=1)
df.drop(leave,axis=1,inplace=True)
df
# 2. binary
binary_encoding([binary[0]], ["야","주"], "night")
binary_encoding([binary[1]], ["남","여"], "at_male")
binary_encoding([binary[2]], ["남","여"], "vt_male")
binary_encoding([binary[3]], ["음주운전","음주상태"], "alch")
binary_encoding([binary[4]], ["안전모","기타불명"], "at_protect")
binary_encoding([binary[5]], ["안전벨트/카시트","안전모"], "vt_protect")
binary_encoding([binary[6]], ["건조","젖음/습기"], "road_stat")
#binary_encoding([binary[7]], ["교차로","교차로아님"], "cross") #삭제
df
def bin_age(x):
    bins= [0,13,16,20,30,40,50,60,70,110]
    labels = [0,13,16,20,30,40,50,60,70]
    return pd.cut(x, bins=bins, labels=labels, right=False)

df['vt_age_bin'] = bin_age(df['연령가해자']).astype('int')
df['at_age_bin'] = bin_age(df['연령피해자']).astype('int')
def make_number(x):  # 숫자만 추출
  if str(x) !="미분류":
    numbers = re.sub(r'[^0-9]', '', str(x))
    if numbers:
      return int(numbers)
    else:
      return 0
  else:
    return -1
df['연령가해자'].apply(make_number)
bin_age(df['연령가해자'].apply(make_number)).astype('int')
df['연령가해자'].apply(make_number)
df['연령가해자'].apply(make_number).value_counts()
df['연령가해자']
def make_number(x):  # 숫자만 추출
  if str(x) !="미분류":
    numbers = re.sub(r'[^0-9]', '', str(x))
    if numbers:
      return int(numbers)
    else:
      return 0
  else:
    return -1
df['연령가해자'] = df['연령가해자'].apply(make_number)
df['연령피해자'] = df['연령피해자'].apply(make_number)
def bin_age(x):
    bins= [0,13,16,20,30,40,50,60,70,110]
    labels = [0,13,16,20,30,40,50,60,70]
    return pd.cut(x, bins=bins, labels=labels, right=False)

df['vt_age_bin'] = bin_age(df['연령가해자']).astype('int')
df['at_age_bin'] = bin_age(df['연령피해자']).astype('int')
def bin_age(x):
    bins= [0,13,16,20,30,40,50,60,70,150]
    labels = [0,13,16,20,30,40,50,60,70]
    return pd.cut(x, bins=bins, labels=labels, right=False)

df['vt_age_bin'] = bin_age(df['연령가해자']).astype('int')
df['at_age_bin'] = bin_age(df['연령피해자']).astype('int')
df
df.drop(['연령가해자','연령피해자'],axis=1)
df.drop(['연령가해자','연령피해자'],axis=1, inplace=True)
df
straight
leftright
roadform
get_ipython().run_line_magic('save', 'categorical_encoding.py 1-300')
