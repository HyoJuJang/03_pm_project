#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


############## datafarme 보여주는 범위 설정 ###################
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.5f}'.format


# In[3]:


################ 경로 설정 및 파일 read #######################
path = 'C:/Users/user/Dropbox/2021_2/intern/project_pm/'
load_path = path + 'pm_data/'  # 데이터 불러오는 경로
save_path = path + 'pm_data/prepro_taas/'  # 전처리 이후 데이터 저장하는 경로
result_path = path + 'pm_results/'


# In[4]:


train1 = pd.read_csv(load_path + 'taas/pm_seoul.csv', encoding="EUC-KR")
train2 = pd.read_csv(load_path + 'taas/2016_2020_가해자자전거사고정보.csv', encoding="EUC-KR")


# In[221]:


d_flat = pd.read_csv(load_path + 'taas/서울자전거_PM결합_경사도추가.csv', encoding='EUC-KR')


# ## 데이터 전처리

# In[222]:


train1.head()


# In[226]:


d_flat[d_flat['Year']!=2016]


# In[224]:


d_flat['SOILSLOPE']


# In[5]:


train1.columns == train2.columns


# In[6]:


train2 = train2[train2['발생지_시도'] == '서울']
#train2.head()


# In[7]:


#################### 속성 전처리 변환 함수 #############################
def make_year(x):  # year 데이터로 변환
    x = str(x)
    return int(x[:4])

def make_month(x):  # month 데이터로 변환
    x = str(x)
    return int(x[4:6])


def make_day(x):  # day 데이터로 변환
    x = str(x)
    return int(x[6:8])

def make_hour(x):  # hour 데이터로 변환
    x = str(x)
    return int(x[-2:])

def make_week(x):  # 요일 데이터로 변환
    week=['일','월','화','수','목','금','토']
    for i in range(0,7):
      if week[i]==x[0:1]:
        return i

def make_acci(x):  # 사고내용 데이터로 변환
    acci=['사','중','경','부'] #사망, 중상, 경상, 부상신고
    for i in range(0,4):
      if acci[i]==x[0:1]:
        return i

def make_acci_B(x):  # 사고내용 대분류 변환
    acci_B=['차대차','차대사람','차량단독'] #차대차, 차대사람, 차량단독
    for i in range(0,3):
      if acci_B[i]==x.split()[0]:
        return i

def make_acci_D(x):  # 사고내용 세부분류 변환
    #acci_B=['차대차','차대사람','차량단독'] #차대차, 차대사람, 차량단독
    #for i in range(0,3):
    return x.split()[2]

def make_road_stat_B(x):  # 노면상태 대분류 변환
    if '포장'==x.split()[0]:
      return 0
    else:
      return 1

def make_road_stat_D(x):  # 노면상태 세부분류 변환
    return x.split()[2]

def make_road_form(x):  # 도로형태 세부분류 변환
    return x.split()[2]

def make_at_car_type(x):  # 가해차종 변환
    at_car_type=['이륜','자전거','원동기','개인형이동수단(PM)'] 
    for i in range(0,4):
      if at_car_type[i]==x:
        return i

def make_gender(x):  # 성별 변환
    at_gender=['남','여']
    if x in at_gender:
        for i in range(0,3):
          if at_gender[i]==x:
            return i
    else:
        return -1

def make_number(x):  # 숫자만 추출
  if str(x) !="미분류":
    numbers = re.sub(r'[^0-9]', '', str(x))
    if numbers:
      return int(numbers)
    else:
      return 0
  else:
    return -1


def make_h_acci(x):  # 운전자 상해정도
    h_acci=['사망','중상','경상','부상신고','상해없음','기타불명','미분류']
    check=1
    for i in range(0,7):
      if h_acci[i]==x:
        check=0
        return i
    if check==1:
      return 7


def make_night(x):
    night=['주','야']
    for i in range(2):
      if night[i]==x[0:1]:
            return i

def alcohol(x):
    al1=['해당 없음','0.030~0.049%','0.05%~0.09%','0.10%~0.14%',
         '0.15%~0.19%','0.20%~0.24%']
    for i in range(8):
        if x in al1:
            if al1[i]==x:
                return i
        else:
            return -1

def alcohol_yes(x):
    al1=['해당 없음','음주운전']
    if x in al1:
        for i in range(2):
          if al1[i]==x:
            return i
    else:
        return -1

def make_yes(x):
    yes=['아니오','예']
    for i in range(2):
      if yes[i]==x:
            return i

def make_linear(x):
    line=['직선','커브ㆍ곡각 ']
    if x!= '기타구역':
        for i in range(2):
            if line[i]==x:
                return i
    else:
        return -1

def make_straight(x):
    line=['직선','우','좌']
    if x!= '기타구역':
        for i in range(3):
            if line[i]==x:
                return i
    else:
        return -1   

def make_flat(x):
    flat=['평지','오르막','내리막']
    if x in flat:
        for i in range(3):
            if flat[i]==x:
                return i
    else:
        return -1   
    
def make_cross(x):
    cross=['교차로아님','교차로']
    for i in range(2):
      if cross[i]==x:
            return i

def make_cross_cnt(x):
    cross=['교차로아님','교차로 - 삼지','교차로 - 사지','교차로 - 오지이상']
    for i in range(4):
      if cross[i]==x:
            return i
        
#####################################################################


# In[8]:


copytrain1 = train1.copy()
pretrain1 = pd.DataFrame()

pretrain1['year'] = copytrain1['TAAS사고관리번호'].apply(make_year)
pretrain1['month'] = copytrain1['TAAS사고관리번호'].apply(make_month)
pretrain1['day'] = copytrain1['TAAS사고관리번호'].apply(make_day)
pretrain1['hour'] = copytrain1['발생일시'].apply(make_hour)
pretrain1['week'] = copytrain1['요일'].apply(make_week)
pretrain1['night'] = copytrain1['주야']
#시군구를 어떻게 활용할까?
pretrain1['accident'] = copytrain1['사고내용'].apply(make_acci)
pretrain1['d_acci'] = copytrain1['사망자수']       ############
pretrain1['s_acci'] = copytrain1['중상자수']
pretrain1['c_acci'] = copytrain1['경상자수']
pretrain1['i_acci'] = copytrain1['부상신고자수']   # 4가지를 EPDO같은 수치로 변환하여 사용할 것인지 ?
pretrain1['acci_case_B'] = copytrain1['사고유형_대분류']  
pretrain1['acci_case_D'] = copytrain1['사고유형_중분류']   # 추후 interaction을 통해 파생변수 생성
pretrain1['law_viol'] = copytrain1['법규위반가해자']
pretrain1['road_stat_B'] = copytrain1['노면상태_대분류']
pretrain1['road_stat_D'] = copytrain1['노면상태']
pretrain1['road_form_B'] = copytrain1['도로형태_대분류']
pretrain1['road_form_D'] = copytrain1['도로형태']
#pretrain1['at_car_type'] = copytrain1['가해운전자 차종'].apply(make_at_car_type)
pretrain1['at_gender'] = copytrain1['성별가해자'].apply(make_gender)
pretrain1['at_age'] = copytrain1['연령가해자'].apply(make_number)
pretrain1['at_acci'] = copytrain1['신체상해정도가해자'].apply(make_h_acci)

pretrain1['vt_car_class'] = copytrain1['차량용도피해자_대분류']
pretrain1['vt_car_type_A'] = copytrain1['차량용도피해자_중분류']
pretrain1['vt_gender'] = copytrain1['성별가해자_1'].apply(make_gender)
pretrain1['vt_age'] = copytrain1['연령피해자'].apply(make_number)
pretrain1['vt_acci'] = copytrain1['신체상해정도피해자'].apply(make_h_acci)

pretrain1['night'] = copytrain1['주야'].apply(make_night)
pretrain1['gu'] = copytrain1['발생지_시군구']
pretrain1['alch'] = copytrain1['음주측정수치가해자_대분류'].apply(alcohol_yes)
pretrain1['alch_cont'] = copytrain1['음주측정수치가해자'].apply(alcohol)

pretrain1['at_protect'] = copytrain1['보호장구가해자']
pretrain1['vt_protect_type'] = copytrain1['보호장구피해자_대분류']
pretrain1['vt_protect'] = copytrain1['보호장구피해자']
pretrain1['at_driving_B'] = copytrain1['행동유형가해자_중분류']
pretrain1['at_driving_D'] = copytrain1['행동유형가해자']
pretrain1['at_acci_part'] = copytrain1['가해자신체상해주부위']
pretrain1['vt_acci_part'] = copytrain1['피해자신체상해주부위']
pretrain1['vt_car_type_B'] = copytrain1['당사자종별피해자']

pretrain1['elder'] = copytrain1['노인보호구역_여부'].apply(make_yes)
pretrain1['child'] = copytrain1['어린이보호구역_여부'].apply(make_yes)
pretrain1['cycle'] = copytrain1['자전거도로_여부'].apply(make_yes)

pretrain1['road_linear'] = copytrain1['도로선형_대분류'].apply(make_linear)
pretrain1['road_straight'] = copytrain1['도로선형_중분류'].apply(make_straight)
pretrain1['road_flat'] = copytrain1['도로선형'].apply(make_flat)

pretrain1['cross'] = copytrain1['교차로형태_대분류'].apply(make_cross)
pretrain1['cross_cnt'] = copytrain1['교차로형태'].apply(make_cross_cnt)
pretrain1['weather'] = copytrain1['기상상태']

pretrain1.info()


# In[9]:


copytrain2 = train2.copy()
pretrain2 = pd.DataFrame()

pretrain2['year'] = copytrain2['TAAS사고관리번호'].apply(make_year)
pretrain2['month'] = copytrain2['TAAS사고관리번호'].apply(make_month)
pretrain2['day'] = copytrain2['TAAS사고관리번호'].apply(make_day)
pretrain2['hour'] = copytrain2['발생일시'].apply(make_hour)
pretrain2['week'] = copytrain2['요일'].apply(make_week)
pretrain2['night'] = copytrain2['주야']
#시군구를 어떻게 활용할까?
pretrain2['accident'] = copytrain2['사고내용'].apply(make_acci)
pretrain2['d_acci'] = copytrain2['사망자수']       ############
pretrain2['s_acci'] = copytrain2['중상자수']
pretrain2['c_acci'] = copytrain2['경상자수']
pretrain2['i_acci'] = copytrain2['부상신고자수']   # 4가지를 EPDO같은 수치로 변환하여 사용할 것인지 ?
pretrain2['acci_case_B'] = copytrain2['사고유형_대분류']  
pretrain2['acci_case_D'] = copytrain2['사고유형_중분류']   # 추후 interaction을 통해 파생변수 생성
pretrain2['law_viol'] = copytrain2['법규위반가해자']
pretrain2['road_stat_B'] = copytrain2['노면상태_대분류']
pretrain2['road_stat_D'] = copytrain2['노면상태']
pretrain2['road_form_B'] = copytrain2['도로형태_대분류']
pretrain2['road_form_D'] = copytrain2['도로형태']
#pretrain2['at_car_type'] = copytrain2['가해운전자 차종'].apply(make_at_car_type)
pretrain2['at_gender'] = copytrain2['성별가해자'].apply(make_gender)
pretrain2['at_age'] = copytrain2['연령가해자'].apply(make_number)
pretrain2['at_acci'] = copytrain2['신체상해정도가해자'].apply(make_h_acci)

pretrain2['vt_car_class'] = copytrain2['차량용도피해자_대분류']
pretrain2['vt_car_type_A'] = copytrain2['차량용도피해자_중분류']
pretrain2['vt_gender'] = copytrain2['성별가해자_1'].apply(make_gender)
pretrain2['vt_age'] = copytrain2['연령피해자'].apply(make_number)
pretrain2['vt_acci'] = copytrain2['신체상해정도피해자'].apply(make_h_acci)

pretrain2['night'] = copytrain2['주야'].apply(make_night)
pretrain2['gu'] = copytrain2['발생지_시군구']
pretrain2['alch'] = copytrain2['음주측정수치가해자_대분류'].apply(alcohol_yes)
pretrain2['alch_cont'] = copytrain2['음주측정수치가해자'].apply(alcohol)

pretrain2['at_protect'] = copytrain2['보호장구가해자']
pretrain2['vt_protect_type'] = copytrain2['보호장구피해자_대분류']
pretrain2['vt_protect'] = copytrain2['보호장구피해자']
pretrain2['at_driving_B'] = copytrain2['행동유형가해자_중분류']
pretrain2['at_driving_D'] = copytrain2['행동유형가해자']
pretrain2['at_acci_part'] = copytrain2['가해자신체상해주부위']
pretrain2['vt_acci_part'] = copytrain2['피해자신체상해주부위']
pretrain2['vt_car_type_B'] = copytrain2['당사자종별피해자']

pretrain2['elder'] = copytrain2['노인보호구역_여부'].apply(make_yes)
pretrain2['child'] = copytrain2['어린이보호구역_여부'].apply(make_yes)
pretrain2['cycle'] = copytrain2['자전거도로_여부'].apply(make_yes)

pretrain2['road_linear'] = copytrain2['도로선형_대분류'].apply(make_linear)
pretrain2['road_straight'] = copytrain2['도로선형_중분류'].apply(make_straight)
pretrain2['road_flat'] = copytrain2['도로선형'].apply(make_flat)

pretrain2['cross'] = copytrain2['교차로형태_대분류'].apply(make_cross)
pretrain2['cross_cnt'] = copytrain2['교차로형태'].apply(make_cross_cnt)
pretrain2['weather'] = copytrain2['기상상태']

pretrain2.info()


# In[10]:


pretrain2 = pretrain2[pretrain2['year'] != 2016]


# In[11]:


pretrain1.describe().iloc[[3,7],:]


# In[12]:


pretrain2.describe().iloc[[3,7],:]


# In[13]:


pretrain1['year'].value_counts()


# In[14]:


pretrain2['year'].value_counts()


# In[15]:


pretrain1.describe()


# In[16]:


pretrain2.describe()


# #### 추가변수

# In[17]:


def make_form_B(x):
    if x == '교차로':
        return 1
    else:
        return 0
def make_driving_B(x):
    if x == '회전관련':
        return 1
    else:
        return 0
def bin_age(x):
    bins= [0,13,16,20,30,40,50,60,70,110]
    labels = [0,13,16,20,30,40,50,60,70]
    return pd.cut(x, bins=bins, labels=labels, right=False)


# In[18]:


pretrain1['crossroad'] = copytrain1['도로형태_대분류'].apply(make_form_B)
pretrain1['driving_B'] = copytrain1['행동유형가해자_중분류'].apply(make_driving_B)
pretrain1['vt_age_bin'] = bin_age(pretrain1['vt_age']).astype('int')
pretrain1['at_age_bin'] = bin_age(pretrain1['at_age']).astype('int')


# In[19]:


plt.hist(pretrain2['at_age'])
plt.show()


# In[20]:


plt.hist(pretrain2[pretrain2['at_age']>85]['at_age'])
plt.show()


# In[21]:


### 100세 이상 삭제
pretrain2 = pretrain2[(pretrain2['at_age']<=100) & (pretrain2['vt_age']<=100)]


# In[22]:


pretrain2['crossroad'] = copytrain2['도로형태_대분류'].apply(make_form_B)
pretrain2['driving_B'] = copytrain2['행동유형가해자_중분류'].apply(make_driving_B)
pretrain2['vt_age_bin'] = bin_age(pretrain2['vt_age']).astype('int')
pretrain2['at_age_bin'] = bin_age(pretrain2['at_age']).astype('int')


# In[26]:


d = pretrain1['d_acci'].sum()
s = pretrain1['s_acci'].sum()
c = pretrain1['c_acci'].sum()
i = pretrain1['i_acci'].sum()


# In[28]:


d = pretrain2['d_acci'].sum()
s = pretrain2['s_acci'].sum()
c = pretrain2['c_acci'].sum()
i = pretrain2['i_acci'].sum()


# In[27]:


(c+i)/(d+s+c+i) #PM


# In[29]:


(c+i)/(d+s+c+i) #자전거


# In[26]:


############## 사고 심각도 변수 생성 : EPDO, ARI ###################
#d_acci, s_acci, c_acci, i_acci의 선형 결합
#2020년 인천지부 수식 참고


# In[30]:


dsci = pretrain1[['d_acci', 's_acci','c_acci','i_acci']]
ksi = pretrain1[['d_acci', 's_acci']]
pretrain1['epdo']= np.dot(dsci, [273, 32, 2, 1])
pretrain1['ari']=np.sqrt(np.square(np.dot(ksi,[1,1])) + np.square(np.dot(dsci,[1,1,1,1])))/4


# In[31]:


pretrain1[['epdo','ari']]


# In[32]:


dsci = pretrain2[['d_acci', 's_acci','c_acci','i_acci']]
ksi = pretrain2[['d_acci', 's_acci']]
pretrain2['epdo']= np.dot(dsci, [273, 32, 2, 1])
pretrain2['ari']=(np.sqrt(np.square(np.dot(ksi,[1,1])) + np.square(np.dot(dsci,[1,1,1,1]))))/4


# In[33]:


pretrain2[['epdo','ari']]


# ### PM + 자전거 데이터프레임

# In[34]:


pretrain1['pm'] = 1
pretrain2['pm'] = 0


# In[35]:


df = pd.concat([pretrain1, pretrain2])
df_org = df.copy()


# In[36]:


df.reset_index(drop=True, inplace=True)


# In[37]:


df.head()


# In[35]:


df['pm'].value_counts()


# In[38]:


col = df.columns


# In[39]:


col_categ = col[df.dtypes == 'object']
col_conti = col[df.dtypes != 'object']
print(col_categ)
print(col_conti)


# In[40]:


df.loc[:,[(df[col].dtype != 'object') for col in df.columns]]


# In[41]:


df.loc[:,[(df[col].dtype == 'object') for col in df.columns]]


# #### 차대차, 차대사람, 차량단독으로 데이터 쪼개고
# #### 마지막으로 변수들 변환(원핫인코딩 포함) 및 제거

# In[42]:


print(df['law_viol'].value_counts())
print('\n')
print(df['road_stat_D'].value_counts())
print('\n')
#print(df['road_form_B'].value_counts())
#print('\n')
print(df['at_protect'].value_counts())
print('\n')


# In[43]:


df['acci_case_B'].value_counts()


# In[44]:


def one_hot_encoding(var, col):
    global df
    enc = OneHotEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(var).toarray())
    for i in range(len(enc_df.columns)):
        enc_df.iloc[:,i] = enc_df.iloc[:,i].astype('int')
    enc_df.columns = col
    df = df.join(enc_df)


# In[45]:


one_hot_encoding(df[['law_viol']],['law_cross','law_etc','law_pedestrian','law_traffic','law_distance','law_safety','law_center'])


# In[46]:


one_hot_encoding(df[['at_protect']],['tm1','at_prot_n','at_prot_y','tm2'])


# In[47]:


one_hot_encoding(df[['at_age_bin']],['at_0','at_13','at_16','at_20','at_30','at_40','at_50','at_60','at_70'])


# In[48]:


one_hot_encoding(df[['vt_age_bin']],['vt_0','vt_13','vt_16','vt_20','vt_30','vt_40','vt_50','vt_60','vt_70'])


# In[49]:


one_hot_encoding(df[['at_gender']],['at_gender_na','at_gender_m','at_gender_f'])


# In[50]:


one_hot_encoding(df[['vt_gender']],['vt_gender_na','vt_gender_m','vt_gender_f'])


# In[51]:


one_hot_encoding(df[['alch_cont']],['alch_na','alch_0','alch_0.030_0.049','alch_0.05_0.09','alch_0.10_0.14','alch_0.15_0.19','alch_0.20_0.24'])


# In[52]:


one_hot_encoding(df[['road_straight']],['straight_na','straight_s','straight_r','straight_l'])


# In[217]:


def one_hot_encoding_cate(var):
    enc = OneHotEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(var).toarray())
    print(enc.categories_)


# In[220]:


one_hot_encoding_cate(df[['at_protect']])


# In[ ]:





# In[53]:


df.head()


# In[54]:


df = df.loc[:,[(df[col].dtype != 'object') for col in df.columns]]


# In[55]:


dt = df.drop(labels=['hour','accident','d_acci','s_acci','c_acci','i_acci','at_age','vt_age','at_age_bin',
                     'vt_age_bin','at_acci','vt_acci','alch','elder','cycle','cross',
                    'vt_gender','at_gender','alch_cont', 'road_linear', 'road_straight', 'road_flat'],axis=1)


# In[56]:


dt.head()


# In[57]:


dt_org = dt.copy()
dt['acci_case_B'] = df_org['acci_case_B']


# In[58]:


dt.columns


# In[60]:


### ari, epdo log 변환
dt['log_ari'] = np.log(dt['ari']+1)
dt['log_epdo'] = np.log(dt['epdo']+1)


# In[61]:


dt = dt.drop(labels=['ari','epdo'],axis=1)


# In[62]:


dt_vv = dt[dt['acci_case_B']=='차대차']
dt_vp = dt[dt['acci_case_B']=='차대사람']
dt_va = dt[dt['acci_case_B']=='차량단독']


# In[63]:


dt_vv = dt_vv.drop(labels='acci_case_B',axis=1)
dt_vp = dt_vp.drop(labels='acci_case_B',axis=1)
dt_va = dt_va.drop(labels='acci_case_B',axis=1)


# In[64]:


dt_vv.head()


# ### ARI

# In[65]:


dt_vv = dt_vv.drop(labels='log_epdo',axis=1)
dt_vp = dt_vp.drop(labels='log_epdo',axis=1)
dt_va = dt_va.drop(labels='log_epdo',axis=1)


# In[66]:


pd.value_counts(dt_vv['pm']).plot.bar()
plt.title('VV pm class histogram')
plt.xlabel('pm')
plt.ylabel('Frequency')
dt_vv['pm'].value_counts()


# In[67]:


pd.value_counts(dt_vp['pm']).plot.bar()
plt.title('VP pm class histogram')
plt.xlabel('pm')
plt.ylabel('Frequency')
dt_vp['pm'].value_counts()


# In[68]:


pd.value_counts(dt_va['pm']).plot.bar()
plt.title('VA pm class histogram')
plt.xlabel('pm')
plt.ylabel('Frequency')
dt_va['pm'].value_counts()


# In[ ]:





# # 분석

# ## (1) VP(차대사람) 분석

# In[69]:


dt_vp_org = dt_vp.copy()
dt_vp.describe()


# In[129]:


dt_vp = dt_vp.drop('child',axis=1)


# In[ ]:





# In[71]:


#sns.set(font_scale=0.8)
dataframe = dt_vp

plt.figure(figsize=(18, 10))
corr = dataframe.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
cut_off = 0.15  # only show cells with abs(correlation) at least this value
extreme_1 = 0.2  # show with a star
extreme_2 = 0.3  # show with a second star
extreme_3 = 0.4  # show with a third star
mask |= np.abs(corr) < cut_off
corr = corr[~mask]  # fill in NaN in the non-desired cells

remove_empty_rows_and_cols = True
if remove_empty_rows_and_cols:
    wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
    wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
    corr = corr.iloc[wanted_cols, wanted_rows]

annot = [[f"{val:.2f}"
          + ('' if abs(val) < extreme_1 else '\n★')  # add one star if abs(val) >= extreme_1
          + ('' if abs(val) < extreme_2 else '★')  # add an extra star if abs(val) >= extreme_2
          + ('' if abs(val) < extreme_3 else '★')  # add yet an extra star if abs(val) >= extreme_3
          for val in row] for row in corr.to_numpy()]
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=annot, fmt='', cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
plt.show()
#plt.savefig('fig1.jpg', dpi=400)


# In[83]:


dt_vp.columns[-5:]


# In[84]:


pt1_col1 = dt_vp.columns[4:10]
pt1_col2 = dt_vp.columns[10:17]
pt1_col3 = dt_vp.columns[-5:]
print(pt1_col1)
print(pt1_col2)
print(pt1_col3)


# In[85]:


pt1 = dt_vp[pt1_col1]
pt2 = dt_vp[pt1_col2]
pt3 = dt_vp[pt1_col3]
pt1 = pd.concat([pt1, dt_vp[['log_ari']]], axis=1)
pt2 = pd.concat([pt2, dt_vp[['pm','log_ari']]], axis=1)
pt3 = pd.concat([pt3, dt_vp[['pm']]], axis=1)


# In[86]:


sns.set(font_scale = 2)
sns.pairplot(pt1, diag_kind='kde')
plt.show()


# In[87]:


sns.set(font_scale = 2)
sns.pairplot(pt2, diag_kind='kde')
plt.show()


# In[88]:


sns.set(font_scale = 2)
sns.pairplot(pt3, diag_kind='kde')
plt.show()


# In[85]:


sns.set(font_scale=2)
plt.figure(figsize=(12,8))
sns.distplot(dt['ari'])
plt.show()


# In[86]:


sns.set(font_scale=2)
plt.figure(figsize=(12,8))
sns.distplot(dt_vp['log_ari'])
plt.show()


# In[89]:


dt_vp['log_ari'].value_counts()


# ## Logistic Regression

# In[130]:


X = dt_vp.drop('pm',axis=1)
y = dt_vp['pm']
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))


# In[131]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[132]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf.fit(X_train, y_train.ravel())


# In[133]:


clf.best_params_


# In[134]:


lr1 = LogisticRegression(C=6, verbose=5)
lr1.fit(X_train, y_train.ravel())


# In[135]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[136]:


sns.set(font_scale=1)


# In[137]:


y_train_pre = lr1.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()


# In[138]:


y_pre = lr1.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pre)

print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()


# In[139]:


tmp = lr1.fit(X_train, y_train.ravel())
y_pred_sample_score = tmp.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[140]:


roc_auc


# In[141]:


lr1.get_params()


# In[142]:


pd.concat([pd.Series(X_train.columns),pd.Series(lr1.coef_[0])], axis=1).sort_values(by=1, key=abs,ascending=False)


# In[ ]:





# In[161]:


X_train_rm_age = X_train.drop(['at_0','at_13','at_16','at_20','at_30','at_40','at_50','at_60','at_70',
                              'vt_0','vt_13','vt_16','vt_20','vt_30','vt_40','vt_50','vt_60','vt_70'],axis=1)
X_train_rm_age_gender = X_train_rm_age.drop(['at_gender_na','at_gender_m','at_gender_f','vt_gender_na','vt_gender_m','vt_gender_f'],axis=1)


# In[178]:


X_test_rm_age = X_test.drop(['at_0','at_13','at_16','at_20','at_30','at_40','at_50','at_60','at_70',
                              'vt_0','vt_13','vt_16','vt_20','vt_30','vt_40','vt_50','vt_60','vt_70'],axis=1)
X_test_rm_age_gender = X_test_rm_age.drop(['at_gender_na','at_gender_m','at_gender_f','vt_gender_na','vt_gender_m','vt_gender_f'],axis=1)


# In[191]:


parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
clf2 = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf2.fit(X_train_rm_age, y_train.ravel())


# In[192]:


clf2.best_estimator_


# In[193]:


lr2 = LogisticRegression(C=6, verbose=5)
lr2.fit(X_train_rm_age, y_train.ravel())


# In[194]:


y_train_pre2 = lr2.predict(X_train_rm_age)


# In[196]:


cnf_matrix_tra2 = confusion_matrix(y_train, y_train_pre2)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra2[1,1]/(cnf_matrix_tra2[1,0]+cnf_matrix_tra2[1,1])))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra2 , classes=class_names, title='Confusion matrix')
plt.show()


# In[197]:


y_pre2 = lr2.predict(X_test_rm_age)
cnf_matrix2 = confusion_matrix(y_test,y_pre2)


# In[198]:


print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix2[1,1]/(cnf_matrix2[1,0]+cnf_matrix2[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix2 , classes=class_names, title='Confusion matrix')
plt.show()


# In[199]:


tmp = lr2.fit(X_train_rm_age, y_train.ravel())
y_pred_sample_score2 = tmp.decision_function(X_test_rm_age)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score2)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[200]:


roc_auc


# In[211]:


pd.concat([pd.Series(X_train_rm_age.columns),pd.Series(lr2.coef_[0])], axis=1).sort_values(by=1, key=abs,ascending=False)


# In[ ]:





# In[ ]:





# In[201]:


parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
clf3 = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf3.fit(X_train_rm_age_gender, y_train.ravel())


# In[202]:


clf3.best_estimator_


# In[203]:


lr3 = LogisticRegression(C=6, verbose=5)
lr3.fit(X_train_rm_age_gender, y_train.ravel())


# In[204]:


y_train_pre3 = lr3.predict(X_train_rm_age_gender)


# In[205]:


cnf_matrix_tra3 = confusion_matrix(y_train, y_train_pre3)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra3[1,1]/(cnf_matrix_tra3[1,0]+cnf_matrix_tra3[1,1])))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra3 , classes=class_names, title='Confusion matrix')
plt.show()


# In[206]:


y_pre3 = lr3.predict(X_test_rm_age_gender)
cnf_matrix3 = confusion_matrix(y_test,y_pre3)


# In[207]:


print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix3[1,1]/(cnf_matrix3[1,0]+cnf_matrix3[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix3 , classes=class_names, title='Confusion matrix')
plt.show()


# In[208]:


tmp = lr3.fit(X_train_rm_age_gender, y_train.ravel())
y_pred_sample_score3 = tmp.decision_function(X_test_rm_age_gender)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score3)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[209]:


roc_auc


# In[215]:


dt_vp[dt_vp['pm']==1]['law_safety'].value_counts()


# In[216]:


dt_vp[dt_vp['pm']==0]['law_safety'].value_counts()


# In[212]:


pd.concat([pd.Series(X_train_rm_age_gender.columns),pd.Series(lr3.coef_[0])], axis=1).sort_values(by=1, key=abs,ascending=False)


# In[ ]:





# In[117]:


dt_vv.to_csv(save_path + "dt_vv.csv", encoding="EUC-KR", index=False)
dt_vp.to_csv(save_path + "dt_vp.csv", encoding="EUC-KR", index=False)
dt_va.to_csv(save_path + "dt_va.csv", encoding="EUC-KR", index=False)


# In[1]:


get_ipython().run_line_magic('save', '-a propre_logistic.py 1-100')

