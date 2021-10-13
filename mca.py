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
from sklearn.preprocessing import OneHotEncoder

#from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')
############## datafarme 보여주는 범위 설정 ###################
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.5f}'.format
################ 경로 설정 및 파일 read #######################
path = 'C:/Users/user/Dropbox/2021_2/intern/project_pm/'
load_path = path + 'pm_data/'  # 데이터 불러오는 경로
save_path = path + 'pm_data/prepro_taas/'  # 전처리 이후 데이터 저장하는 경로
result_path = path + 'pm_results/'
train1 = pd.read_csv(load_path + 'taas/pm_seoul.csv', encoding="EUC-KR")
train2 = pd.read_csv(load_path + 'taas/2016_2020_가해자자전거사고정보.csv', encoding="EUC-KR")
d_flat = pd.read_csv(load_path + 'flat_sidewalk/중첩.csv', encoding='EUC-KR')
def make_year(x):  # year 데이터로 변환
    x = str(x)
    return int(x[:4])

def make_sidewalks(x):
    if pd.isna(x):
        return 0
    else:
        return 1

def make_number(x):  # 숫자만 추출
  if str(x) !="미분류":
    numbers = re.sub(r'[^0-9]', '', str(x))
    if numbers:
      return int(numbers)
    else:
      return 0
  else:
    return -1

def make_pm(x):
    if x == 1:
        return 1
    else:
        return 0
    
def bin_age(x):
    bins= [0,13,16,20,30,40,50,60,70,110]
    labels = [0,13,16,20,30,40,50,60,70]
    return pd.cut(x, bins=bins, labels=labels, right=False)
# 서울만
train2 = train2[train2['발생지_시도'] == '서울']
# 2016년 제거
train2['year'] = train2['TAAS사고관리번호'].apply(make_year)
train2 = train2[train2['year']!=2016]
train2 = train2.drop('year',axis=1)
d_flat = d_flat[d_flat['Year']!=2016]
### 데이터 프레임 합치기
df = pd.concat([train1, train2]).reset_index(drop=True)
d_flat.reset_index(drop=True, inplace=True)
dt = pd.concat([df, d_flat[['Class','SOILSLOPE','yes_count']]],axis=1)
dt.columns
dt = dt.drop(['TAAS사고관리번호', '발생일시', '발생지_시도', '발생지_시군구', 'X_POINT', 'Y_POINT', 
         '요일','사고내용', '사고유형_중분류', '사고유형', '음주측정수치가해자', 
         '보호장구가해자_대분류','보호장구피해자_대분류', '행동유형가해자_대분류',
       '행동유형가해자_중분류', '신체상해정도가해자', '신체상해정도피해자', '가해자신체상해주부위',
       '피해자신체상해주부위', '당사자종별가해자_대분류', '당사자종별가해자', '당사자종별피해자_대분류', '당사자종별피해자',
       '차량용도가해자_대분류', '차량용도가해자_중분류', '차량용도가해자', 
         '차량용도피해자_대분류', '차량용도피해자_중분류', '차량용도피해자', 
         '도로종류', '도로형태', '노인보호구역_여부', '어린이보호구역_여부',
       '자전거도로_여부', '도로선형_대분류', '도로선형', '기상상태', '노면상태_대분류', '노면상태',
         '교차로형태', '사망자수', '중상자수', '경상자수', '부상신고자수', 
        ], axis=1)
dt['차도'] = dt['yes_count'].apply(make_sidewalks)
dt = dt.drop('yes_count',axis=1)
dt['연령가해자'] = dt['연령가해자'].apply(make_number)
dt['연령피해자'] = dt['연령피해자'].apply(make_number)

dt['연령가해자'] = bin_age(dt['연령가해자'])
dt['연령피해자'] = bin_age(dt['연령피해자'])
dt['Class'] = dt['Class'].apply(make_pm)
dt.columns = ['night','acci_case_B','at_gender','vt_gender','at_age_bin','vt_age_bin',
              'alch','at_protect','vt_protect','law_viol','at_driving_B',
             'road_linear','road_straight','cross','pm','slope','road']
dt
#!pip install prince
import prince
dt1 = dt.drop('pm',axis=1)
pm_vv = dt[(dt['pm']==1) & (dt['acci_case_B']=='차대차')]
pm_vp = dt[(dt['pm']==1) & (dt['acci_case_B']=='차대사람')]
pm_va = dt[(dt['pm']==1) & (dt['acci_case_B']=='차량단독')]

cy_vv = dt[(dt['pm']==0) & (dt['acci_case_B']=='차대차')]
cy_vp = dt[(dt['pm']==0) & (dt['acci_case_B']=='차대사람')]
cy_va = dt[(dt['pm']==0) & (dt['acci_case_B']=='차량단독')]

d_vv = dt[dt['acci_case_B']=='차대차']
d_vp = dt[dt['acci_case_B']=='차대사람']
d_va = dt[dt['acci_case_B']=='차량단독']
data = pm_vv.drop(['pm','at_age_bin','vt_age_bin','acci_case_B'],axis=1)

mca = prince.MCA(benzecri=True)
mca = mca.fit(data)
mca_t = mca.transform(data)
print(mca_t)
print('\n')
mca.column_coordinates(data)
ax = mca.plot_coordinates(
    X=data,
    ax=None,
    figsize=(6, 6),
    show_row_points=True,
    row_points_size=10,
    show_row_labels=False,
    show_column_points=True,
    column_points_size=30,
    show_column_labels=False,
    legend_n_cols=1
    )
import matplotlib.pyplot as plt

# 폰트 세팅
font = {'family' : 'NanumGothic',
        'weight' : 'bold',
        'size'   : 11}

plt.rc('font', **font)

# 시각화
ax = mca.plot_coordinates(X = data, figsize=(10, 10), show_column_labels=True)
ax.set_title("상응분석", fontsize = 24)
plt.savefig('pm_vv_Ben.png', dpi=300)
data = d_vv.drop(['at_age_bin','vt_age_bin','acci_case_B'],axis=1)

mca = prince.MCA()
mca = mca.fit(data)
mca_t = mca.transform(data)
print(mca_t)
print('\n')
mca.column_coordinates(data)
ax = mca.plot_coordinates(
    X=data,
    ax=None,
    figsize=(6, 6),
    show_row_points=True,
    row_points_size=10,
    show_row_labels=False,
    show_column_points=True,
    column_points_size=30,
    show_column_labels=False,
    legend_n_cols=1
    )
import matplotlib.pyplot as plt

# 폰트 세팅
font = {'family' : 'NanumGothic',
        'weight' : 'bold',
        'size'   : 11}

plt.rc('font', **font)

# 시각화
ax = mca.plot_coordinates(X = data, figsize=(10, 10), show_column_labels=True)
ax.set_title("상응분석", fontsize = 24)
data = pm_vv.drop(['pm','at_age_bin','vt_age_bin','acci_case_B'],axis=1)

mca = prince.MCA(benzecri=True)
mca = mca.fit(data)
mca_t = mca.transform(data)
print(mca_t)
print('\n')
mca.column_coordinates(data)
ax = mca.plot_coordinates(
    X=data,
    ax=None,
    figsize=(6, 6),
    show_row_points=True,
    row_points_size=10,
    show_row_labels=False,
    show_column_points=True,
    column_points_size=30,
    show_column_labels=False,
    legend_n_cols=1
    )
# 폰트 세팅
font = {'family' : 'NanumGothic',
        'weight' : 'bold',
        'size'   : 11}

plt.rc('font', **font)

# 시각화
ax = mca.plot_coordinates(X = data, figsize=(10, 10), show_column_labels=True)
ax.set_title("상응분석", fontsize = 24)
plt.savefig('pm_vv.png', dpi=300)
get_ipython().run_line_magic('save', 'mca.py 1-300')
