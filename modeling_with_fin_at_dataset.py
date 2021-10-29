# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

############## datafarme 보여주는 범위 설정 ###################
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.4f}'.format

path = 'C:/Users/user/Dropbox/2021_2/intern/project_pm/'
load_path = path + 'pm_data/'  # 데이터 불러오는 경로
save_path = path + 'pm_data/prepro_taas/'  # 전처리 이후 데이터 저장하는 경로
result_path = path + 'pm_results/'
df = pd.read_csv(load_path + 'taas/2017_2020_서울_가해자PM_자전거_이륜차사고정보통합(외부데이터통합ver).csv', encoding='EUC-KR')
cate = pd.read_csv(load_path + 'taas/catego3.csv', encoding='EUC-KR')

df.shape
df_ljc = df.copy()
df_pmbycmoto=df_ljc[df_ljc['당사자종별가해자_대분류']!='원동기장치자전거']
df_pmbycmoto.reset_index(inplace=True, drop=True)
cate
dt=df_pmbycmoto.drop([x for x in cate[cate['method']=='delete'].loc[:,'column']],axis=1,inplace=False)
dt.drop(['index'],axis=1,inplace=True)
dt.drop(['당사자종별가해자_대분류'],axis=1,inplace=True)
dt.reset_index(inplace=True, drop=True)
dt_y=dt['Class']
dt_X=dt.drop(['Class'],axis=1,inplace=False)
dt_X

train_X, test_X, train_y, test_y = train_test_split(dt_X,dt_y,test_size=0.2, random_state=0,stratify = dt_y)
train_X.reset_index(inplace=True,drop=True)
test_X.reset_index(inplace=True,drop=True)
train_y.reset_index(inplace=True,drop=True)
test_y.reset_index(inplace=True,drop=True)
cate['method'].unique()

dummy = [x for x in cate[cate['method']=='dummy'].iloc[:,0]]
binary = [x for x in cate[cate['method']=='binary'].iloc[:,0]]
ordinal = [x for x in cate[cate['method']=='ordinal'].iloc[:,0]]
leave = [x for x in cate[cate['method']=='LeaveOneOut'].iloc[:,0]]
binning = [x for x in cate[cate['method']=='binning'].iloc[:,0]]
numeric = [x for x in cate[cate['method']=='numeric'].iloc[:,0]]

def save_df(data):
    global save_pre
    if save_pre.empty:
        save_pre=data
    else:
        save_pre=pd.concat([save_pre,data],axis=1)
def save_df_test(data):
    global save_pre_t
    if save_pre_t.empty:
        save_pre_t=data
    else:
        save_pre_t=pd.concat([save_pre_t,data],axis=1)
save_pre=pd.DataFrame()
save_pre_t=pd.DataFrame()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
def dummy_encoding(var, atr, newcol):
    df_temp = pd.DataFrame()
    odn = pd.Categorical(var, categories = atr, ordered = False)
    odn_v = pd.DataFrame(odn.codes)
    enc = OneHotEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(odn_v).toarray())

    enc_df.columns = newcol
    df_temp= enc_df.iloc[:,1:] # 앞에 변수 빼줌으로써, dummy 변환  
    return df_temp
def binary_encoding(data, atr_t,atr_f, newname): # 변수, 1로 변환할 변수명, 새로운 변수 이름
    bn = []  
    for index, row in data.iterrows():
        if row[0] in atr_t:
            bn.append(1)
        elif row[0] in atr_f:
            bn.append(0)
        else:
            bn.append(-1)

    bv = pd.DataFrame(bn)
    bv.columns = newname
    return bv
def make_ordinal(data, m_list,col,w):
    series=data.reset_index(drop=True).copy()
    for i in range(len(m_list)):
        num=0
        for index, row in series.iterrows():
            if row[0] in m_list[i]:
                series.loc[num]=w[i]
            num=num+1
    series.columns = col
    return series
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
class TargetEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols=None):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
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
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
class TargetEncoderLOO(TargetEncoder):
    
    def __init__(self, cols=None):
        self.cols = cols
        
    def fit(self, X, y):
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
        return self.fit(X, y).transform(X, y)
def LOOencoding(X,y):
    y=y.astype(str)
    enc=ce.OneHotEncoder().fit(y)
    y_onehot=enc.transform(y)
    class_names=y_onehot.columns
    
    X_new=pd.DataFrame()
    name=['pm','byc','motor']
    enc_save=[]
    for i in range(len(class_names)):
        cl = class_names[i]
        enc=TargetEncoderLOO()
        enc_save.append(enc)
        enc.fit(X,y_onehot[cl])
        temp=enc.fit_transform(X,y_onehot[cl])
        temp.columns=[str(x)+'_'+str(name[i]) for x in temp.columns]
        X_new=pd.concat([X_new,temp],axis=1)
    return X_new, enc_save
def TargetEncoding(X,y):
    y=y.astype(str)
    enc=ce.OneHotEncoder().fit(y)
    y_onehot=enc.transform(y)
    class_names=y_onehot.columns
    
    X_new=pd.DataFrame()
    name=['pm','byc','motor']
    for i in range(len(class_names)):
        cl = class_names[i]
        enc=TargetEncoder()
        temp=enc.fit_transform(X,y_onehot[cl])
        temp.columns=[str(x)+'_'+str(name[i]) for x in temp.columns]
        X_new=pd.concat([X_new,temp],axis=1)
    return X_new
def make_mean(data,col):
    temp = data.copy()
    y=data[data[col]!=-1].mean()
    
    for index,row in data.iterrows():
        if row[0] == -1:
            temp.loc[index]=y
            
    return temp
    
    
dt_week_train=dummy_encoding(train_X['요일'],['일','월','화','수','목','금','토'],["sun","mon","tue","wed","thu","fri","sat"])
save_df(dt_week_train)

dt_week_test=dummy_encoding(test_X['요일'],['일','월','화','수','목','금','토'],["sun","mon","tue","wed","thu","fri","sat"])
save_df_test(dt_week_test)
df_slope_train=dummy_encoding(train_X['SOILSLOPE'],['0-2%','2-7%','7-15%','15-30%','기타','30-60%','60-100%'],["slope_1","slope_2","slope_3","slope_4","slope_etc","slope_5","slope_6"])
save_df(df_slope_train)

df_slope_test=dummy_encoding(test_X['SOILSLOPE'],['0-2%','2-7%','7-15%','15-30%','기타','30-60%','60-100%'],["slope_1","slope_2","slope_3","slope_4","slope_etc","slope_5","slope_6"])
save_df_test(df_slope_test)
dt_night_train=binary_encoding(train_X[['주야']],['야'],['주'],['night'])
save_df(dt_night_train)

dt_night_test=binary_encoding(test_X[['주야']],['야'],['주'],['night'])
save_df_test(dt_night_test)
dt_at_gender_train=binary_encoding(train_X[['성별가해자']],['남'],['여'],['at_gender'])
save_df(dt_at_gender_train)

dt_at_gender_test=binary_encoding(test_X[['성별가해자']],['남'],['여'],['at_gender'])
save_df_test(dt_at_gender_test)
dt_vt_gender_train=binary_encoding(train_X[['성별가해자_1']],['남'],['여'],['vt_gender'])
save_df(dt_vt_gender_train)

dt_vt_gender_test=binary_encoding(test_X[['성별가해자_1']],['남'],['여'],['vt_gender'])
save_df_test(dt_vt_gender_test)
dt_alch_train=binary_encoding(train_X[['음주측정수치가해자_대분류']],['음주운전','음주상태'],['해당 없음'],['alch'])
df_alch_mean_train = make_mean(dt_alch_train,dt_alch_train.columns)
save_df(df_alch_mean_train)

dt_alch_test=binary_encoding(test_X[['음주측정수치가해자_대분류']],['음주운전','음주상태'],['해당 없음'],['alch'])
df_alch_mean_test = make_mean(dt_alch_test,dt_alch_test.columns)
save_df_test(df_alch_mean_test)
dt_at_prot_train=binary_encoding(train_X[['보호장구가해자']],['착용'],['미착용'],['at_prot'])
dt_at_prot_mean_train = make_mean(dt_at_prot_train,dt_at_prot_train.columns)
save_df(dt_at_prot_mean_train)

dt_at_prot_test=binary_encoding(test_X[['보호장구가해자']],['착용'],['미착용'],['at_prot'])
dt_at_prot_mean_test = make_mean(dt_at_prot_test,dt_at_prot_test.columns)
save_df_test(dt_at_prot_mean_test)
dt_vt_prot_train=binary_encoding(train_X[['보호장구피해자']],['착용','없음'],['미착용','보행자'],['vt_prot'])
dt_vt_prot_mean_train = make_mean(dt_vt_prot_train,dt_vt_prot_train.columns)
save_df(dt_vt_prot_mean_train)

dt_vt_prot_test=binary_encoding(test_X[['보호장구피해자']],['착용','없음'],['미착용','보행자'],['vt_prot'])
dt_vt_prot_mean_test = make_mean(dt_vt_prot_test,dt_vt_prot_test.columns)
save_df_test(dt_vt_prot_mean_test)
dt_road_stat_train=binary_encoding(train_X[['노면상태']],['건조'],['젖음/습기','기타','서리/결빙','적설','침수','해빙'],['road_stat'])
save_df(dt_road_stat_train)

dt_road_stat_test=binary_encoding(test_X[['노면상태']],['건조'],['젖음/습기','기타','서리/결빙','적설','침수','해빙'],['road_stat'])
save_df_test(dt_road_stat_test)
save_pre
save_pre_t
dt_accident_train= make_ordinal(train_X[['사고내용']],[['부상신고'],['경상'],['중상'],['사망']],['accident'],[0.25,0.5,0.75,1])
save_df(dt_accident_train)

dt_accident_test= make_ordinal(test_X[['사고내용']],[['부상신고'],['경상'],['중상'],['사망']],['accident'],[0.25,0.5,0.75,1])
save_df_test(dt_accident_test)
save_pre
save_pre_t

dt_at_driving_straight_train=binary_encoding(train_X[['행동유형가해자_중분류']],
                                          ['직진관련'],['주정차중','후진 중','주행 중 대기','불명','기타','회전관련'],
                                          ['at_driving_straight'])
save_df(dt_at_driving_straight_train)

dt_at_driving_straight_test=binary_encoding(test_X[['행동유형가해자_중분류']],
                                         ['직진관련'],['주정차중','후진 중','주행 중 대기','불명','기타','회전관련'],
                                         ['at_driving_straight'])
save_df_test(dt_at_driving_straight_test)
dt_at_driving_leftright_train=binary_encoding(train_X[['행동유형가해자_중분류']],
                                          ['회전관련'],['주정차중','후진 중','주행 중 대기','불명','기타','직진관련'],
                                          ['at_driving_leftright'])
save_df(dt_at_driving_leftright_train)

dt_at_driving_leftright_test=binary_encoding(test_X[['행동유형가해자_중분류']],
                                         ['회전관련'],['주정차중','후진 중','주행 중 대기','불명','기타','직진관련'],
                                         ['at_driving_leftright'])
save_df_test(dt_at_driving_leftright_test)
dt_road_form_train= make_ordinal(train_X[['도로형태_대분류']],[['단일로'],['교차로'],['기타','불명','주차장']]
                                 ,['road_form_B'],[0, 1,-1])
save_df(dt_road_form_train)

dt_road_form_test= make_ordinal(test_X[['도로형태_대분류']],[['단일로'],['교차로'],['기타','불명','주차장']]
                                 ,['road_form_B'],[0, 1,-1])
save_df_test(dt_road_form_test)

def bin_age(x):
    bins= [0,13,16,20,30,40,50,60,70,98,150]
    labels = [0,13,16,20,30,40,50,60,70,-1]
    return pd.cut(x, bins=bins, labels=labels, right=False)
def make_number(x):  # 숫자만 추출
    if str(x) != "미분류":
        numbers = re.sub(r'[^0-9]', '', str(x))
        if numbers:
            return int(numbers)
        else:
            return 0
    else:
        return -1

dt_at_age_train= train_X['연령가해자'].apply(make_number)
dt_at_age_train = bin_age(dt_at_age_train).astype('int')
dt_at_age_train_temp=pd.DataFrame(dt_at_age_train)
dt_at_age_train_temp.columns=['at_age']
dt_at_age_train_temp.reset_index(inplace=True,drop=True)
save_df(dt_at_age_train_temp)

dt_at_age_test= test_X['연령가해자'].apply(make_number)
dt_at_age_test = bin_age(dt_at_age_test).astype('int')
dt_at_age_test_temp=pd.DataFrame(dt_at_age_test)
dt_at_age_test_temp.columns=['at_age']
dt_at_age_test_temp.reset_index(inplace=True,drop=True)

save_df_test(dt_at_age_test_temp)
dt_vt_age_train= train_X['연령피해자'].apply(make_number)
dt_vt_age_train = bin_age(dt_vt_age_train).astype('int')
dt_vt_age_train_temp=pd.DataFrame(dt_vt_age_train)
dt_vt_age_train_temp.columns=['vt_age']
dt_vt_age_train_temp.reset_index(inplace=True,drop=True)

save_df(dt_vt_age_train_temp)

dt_vt_age_test= test_X['연령피해자'].apply(make_number)
dt_vt_age_test = bin_age(dt_vt_age_test).astype('int')
dt_vt_age_test_temp=pd.DataFrame(dt_vt_age_test)
dt_vt_age_test_temp.columns=['vt_age']
dt_vt_age_test_temp.reset_index(inplace=True,drop=True)

save_df_test(dt_vt_age_test_temp)
save_pre
save_pre_t

dt_acci_M,LOOenc_1=LOOencoding(train_X[['사고유형_중분류']],train_y)
dt_acci_M.columns=['acci_M_motor','acci_M_byc','acci_M_pm']
save_df(dt_acci_M)

X_test_new =pd.DataFrame()
for i in LOOenc_1:
    dt_acci_M_test= i.transform(test_X[['사고유형_중분류']])
    X_test_new=pd.concat([X_test_new,dt_acci_M_test],axis=1)
X_test_new.columns=['acci_M_motor','acci_M_byc','acci_M_pm']
save_df_test(X_test_new)

dt_law_viol,LOOenc_2=LOOencoding(train_X[['법규위반가해자']],train_y)
dt_law_viol.columns=['law_viol_motor','law_viol_byc','law_viol_pm']
save_df(dt_law_viol)

X_test_new =pd.DataFrame()
for i in LOOenc_2:
    dt_law_viol_test= i.transform(test_X[['법규위반가해자']])
    X_test_new=pd.concat([X_test_new,dt_law_viol_test],axis=1)
X_test_new.columns=['law_viol_motor','law_viol_byc','law_viol_pm']
save_df_test(X_test_new)
dt_at_activ, LOOenc_3=LOOencoding(train_X[['행동유형가해자']],train_y)
dt_at_activ.columns=['at_activ_motor','at_activ_byc','at_activ_pm']
save_df(dt_at_activ)
X_test_new =pd.DataFrame()

for i in LOOenc_3:
    dt_at_activ_test= i.transform(test_X[['행동유형가해자']])
    X_test_new=pd.concat([X_test_new,dt_at_activ_test],axis=1)
X_test_new.columns=['at_activ_motor','at_activ_byc','at_activ_pm']
save_df_test(X_test_new)
dt_road_form, LOOenc_4=LOOencoding(train_X[['도로형태']],train_y)
dt_road_form.columns=['road_form_motor','road_form_byc','road_form_pm']
save_df(dt_road_form)
X_test_new =pd.DataFrame()

for i in LOOenc_4:
    dt_road_form_test= i.transform(test_X[['도로형태']])
    X_test_new=pd.concat([X_test_new,dt_road_form_test],axis=1)
X_test_new.columns=['road_form_motor','road_form_byc','road_form_pm']
save_df_test(X_test_new)
dt_road_straight_M, LOOenc_5=LOOencoding(train_X[['도로선형_중분류']],train_y)
dt_road_straight_M.columns=['road_straight_M_motor','road_straight_M_byc','road_straight_M_pm']
save_df(dt_road_straight_M)
X_test_new =pd.DataFrame()

for i in LOOenc_5:
    dt_road_straight_M_test= i.transform(test_X[['도로선형_중분류']])
    X_test_new=pd.concat([X_test_new,dt_road_straight_M_test],axis=1)
X_test_new.columns=['road_straight_M_motor','road_straight_M_byc','road_straight_M_pm']
save_df_test(X_test_new)
dt_road_straight, LOOenc_6=LOOencoding(train_X[['도로선형']],train_y)
dt_road_straight.columns=['road_straight_motor','road_straight_byc','road_straight_pm']
save_df(dt_road_straight)
X_test_new =pd.DataFrame()

for i in LOOenc_6:
    dt_road_straight_test= i.transform(test_X[['도로선형']])
    X_test_new=pd.concat([X_test_new,dt_road_straight_test],axis=1)
X_test_new.columns=['road_straight_motor','road_straight_byc','road_straight_pm']
save_df_test(X_test_new)
dt_cross_form, LOOenc_7=LOOencoding(train_X[['교차로형태']],train_y)
dt_cross_form.columns=['cross_form_motor','cross_form_byc','cross_form_pm']
save_df(dt_cross_form)
dt_cross_form
X_test_new
X_test_new =pd.DataFrame()

for i in LOOenc_7:
    dt_cross_form_test= i.transform(test_X[['교차로형태']])
    X_test_new=pd.concat([X_test_new,dt_cross_form_test],axis=1)
X_test_new.columns=['cross_form_motor','cross_form_byc','cross_form_pm']
save_df_test(X_test_new)
save_pre
save_pre_t
dt_another_train=train_X.loc[:,['schoolA','silverA','bycA','road_num','busstop_n','tree_n','subway_n']]
save_df(dt_another_train)

dt_another_test=test_X.loc[:,['schoolA','silverA','bycA','road_num','busstop_n','tree_n','subway_n']]
save_df_test(dt_another_test)

save_pre_t
df_try_train=save_pre.copy()
df_try_test=save_pre_t.copy()
df_try_train['group']=train_X['사고유형_대분류']
df_try_train['class']=train_y
df_try_train2 = df_try_train.copy()
df_try_train2.loc[((df_try_train2['road_form_B']== -1) & (df_try_train2['road_num']>= 2)), 'road_form_B'] = 1
df_try_train2.loc[((df_try_train2['road_form_B']== -1) & (df_try_train2['road_num']< 2)), 'road_form_B'] = 0
df_try_test2 = df_try_test.copy()
df_try_test2.loc[((df_try_test2['road_form_B']== -1) & (df_try_test2['road_num']>= 2)), 'road_form_B'] = 1
df_try_test2.loc[((df_try_test2['road_form_B']== -1) & (df_try_test2['road_num']< 2)), 'road_form_B'] = 0
df_try_train = df_try_train2[df_try_train2['at_age'] != -1]
df_try_test = df_try_test2[df_try_test2['at_age'] != -1]

df_vv_train=df_try_train[df_try_train['group']=='차대차']
df_vp_train=df_try_train[df_try_train['group']=='차대사람']
df_va_train=df_try_train[df_try_train['group']=='차량단독']
df_vv_train.drop(['group'],inplace=True,axis=1)
df_vp_train.drop(['group'],inplace=True,axis=1)
df_va_train.drop(['group'],inplace=True,axis=1)
df_try_test['group']=test_X['사고유형_대분류']
df_try_test['class']=test_y

df_vv_test=df_try_test[df_try_test['group']=='차대차']
df_vp_test=df_try_test[df_try_test['group']=='차대사람']
df_va_test=df_try_test[df_try_test['group']=='차량단독']
df_vv_test.drop(['group'],inplace=True,axis=1)
df_vp_test.drop(['group'],inplace=True,axis=1)
df_va_test.drop(['group'],inplace=True,axis=1)

# df_vv_train.to_csv(save_path + "fin/train_vv.csv", encoding="EUC-KR", index=False)
# df_vp_train.to_csv(save_path + "fin/train_vp.csv", encoding="EUC-KR", index=False)
# df_va_train.to_csv(save_path + "fin/train_va.csv", encoding="EUC-KR", index=False)
# df_vv_test.to_csv(save_path + "fin/test_vv.csv", encoding="EUC-KR", index=False)
# df_vp_test.to_csv(save_path + "fin/test_vp.csv", encoding="EUC-KR", index=False)
# df_va_test.to_csv(save_path + "fin/test_va.csv", encoding="EUC-KR", index=False)

df_vv_test2= df_vv_test.copy()
df_vv_train2 = df_vv_train.copy()
df_vv_train = df_vv_train[df_vv_train['at_gender'] != -1]
df_vv_test = df_vv_test[df_vv_test['at_gender'] != -1]
def make_mean2(data,col):
    temp = data.copy()
    y=data[data[col]!=-1][col].mean()
    
    for index,row in data[col].iteritems():
        if row == -1:
            temp.loc[index]=y
            
    return temp

df_vv_train['vt_gender'] = make_mean2(df_vv_train,'vt_gender')['vt_gender']
df_vv_test['vt_gender'] = make_mean2(df_vv_test,'vt_gender')['vt_gender']
# label_test_y=df_vv_test['class']
# df_vv_test.drop(['class'],inplace=True,axis=1)

df_vv_test_drop = df_vv_test.drop(['accident','at_age','vt_age','at_gender', 'vt_gender',
                 'at_prot', 'vt_prot', 'alch', 
                'mon','tue','wed','thu','fri','sat'],axis=1)
df_vv_train_drop = df_vv_train.drop(['accident','at_age','vt_age','at_gender', 'vt_gender',
                 'at_prot', 'vt_prot', 'alch', 
                'mon','tue','wed','thu','fri','sat'],axis=1)
df_vv_test = df_vv_test.drop(['acci_M_pm', 'law_viol_pm', 'at_activ_pm', 
                              'road_form_pm', 'road_straight_M_pm', 'road_straight_pm', 'cross_form_pm'],axis=1)
df_vv_train = df_vv_train.drop(['acci_M_pm', 'law_viol_pm', 'at_activ_pm', 
                              'road_form_pm', 'road_straight_M_pm', 'road_straight_pm', 'cross_form_pm'],axis=1)
df_vv_test_drop = df_vv_test_drop.drop(['acci_M_pm', 'law_viol_pm', 'at_activ_pm', 
                              'road_form_pm', 'road_straight_M_pm', 'road_straight_pm', 'cross_form_pm'],axis=1)
df_vv_train_drop = df_vv_train_drop.drop(['acci_M_pm', 'law_viol_pm', 'at_activ_pm', 
                              'road_form_pm', 'road_straight_M_pm', 'road_straight_pm', 'cross_form_pm'],axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

standardscaler = StandardScaler()
standardscaler.fit(df_vv_train.iloc[:,25:39])
stnd_train_loo = pd.DataFrame(standardscaler.transform(df_vv_train.iloc[:,25:39]), columns=df_vv_train.iloc[:,25:39].columns)
stnd_test_loo = pd.DataFrame(standardscaler.transform(df_vv_test.iloc[:,25:39]), columns=df_vv_train.iloc[:,25:39].columns)
for i in range(7):
    pca = PCA()
    principalComponents = pca.fit_transform(stnd_train_loo.iloc[:,(i*2):(2+i*2)])
    print(pca.explained_variance_ratio_)
# data1 = train, data2 = test

def pca_for_loo(data1, data2):
    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)    
    col = ['acci_M', 'law_viol', 'at_activ', 
        'road_form', 'road_straight_M', 'road_straight', 'cross_form']
    for i in range(7):
        pca = PCA()
        principalComponents1 = pca.fit_transform(stnd_train_loo.iloc[:,(i*2):(2+i*2)])
        data1[col[i]] = pd.DataFrame(principalComponents1).iloc[:,0]
        principalComponents2 = pca.transform(stnd_test_loo.iloc[:,(i*2):(2+i*2)])
        data2[col[i]] = pd.DataFrame(principalComponents2).iloc[:,0]
    return data1, data2

df_vv_train_loo = pca_for_loo(df_vv_train, df_vv_test)[0]
df_vv_test_loo = pca_for_loo(df_vv_train, df_vv_test)[1]

df_vv_train_drop_loo = pca_for_loo(df_vv_train_drop, df_vv_test_drop)[0]
df_vv_test_drop_loo = pca_for_loo(df_vv_train_drop, df_vv_test_drop)[1]
df_vv_train_loo.drop(list(df_vv_train_loo.filter(regex = '._motor')), axis = 1, inplace=True)
df_vv_train_loo.drop(list(df_vv_train_loo.filter(regex = '._byc')), axis = 1, inplace=True)
df_vv_test_loo.drop(list(df_vv_test_loo.filter(regex = '._motor')), axis = 1, inplace=True)
df_vv_test_loo.drop(list(df_vv_test_loo.filter(regex = '._byc')), axis = 1, inplace=True)
df_vv_train_drop_loo.drop(list(df_vv_train_drop_loo.filter(regex = '._motor')), axis = 1, inplace=True)
df_vv_test_drop_loo.drop(list(df_vv_test_drop_loo.filter(regex = '._motor')), axis = 1, inplace=True)
df_vv_train_drop_loo.drop(list(df_vv_train_drop_loo.filter(regex = '._byc')), axis = 1, inplace=True)
df_vv_test_drop_loo.drop(list(df_vv_test_drop_loo.filter(regex = '._byc')), axis = 1, inplace=True)

def make_interaction(dt_vv2):
    col_driving = np.array(['at_driving_straight','at_driving_leftright',
                            'law_viol',
                           'at_activ'])
    col_env = np.array(['road_form', 'road_straight_M',
                        'cross_form', 'schoolA', 'silverA',
                        'bycA', 'road_num', 'busstop_n',
                        'tree_n','subway_n'])
    col_slope = dt_vv2.columns[dt_vv2.columns.str.startswith('slope')]

    for col1 in col_driving:
        for col2 in col_env:
            dt_vv2[col2 + '*' + col1] = dt_vv2[col1].mul(dt_vv2[col2])

    for col1 in col_driving:
        for col2 in col_slope:
            dt_vv2[col2 + '*' + col1] = dt_vv2[col1].mul(dt_vv2[col2])
    return dt_vv2

df_vv_train_loo_inter = make_interaction(df_vv_train_loo)
df_vv_test_loo_inter = make_interaction(df_vv_test_loo)
df_vv_train_drop_loo_inter = make_interaction(df_vv_train_drop_loo)
df_vv_test_drop_loo_inter = make_interaction(df_vv_test_drop_loo)
label_test_y=df_vv_test_loo_inter['class']
df_vv_train_loo_inter_tmp = df_vv_train_loo_inter.drop('class',axis=1)
df_vv_test_loo_inter.drop('class',axis=1, inplace=True)
df_vv_train_drop_loo_inter_tmp = df_vv_train_drop_loo_inter.drop('class',axis=1)
df_vv_test_drop_loo_inter.drop('class',axis=1, inplace=True)

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(df_vv_train_loo_inter_tmp)
df_vv_train_loo_inter_minmax = pd.DataFrame(minMaxScaler.transform(df_vv_train_loo_inter_tmp))
df_vv_test_loo_inter_minmax = pd.DataFrame(minMaxScaler.transform(df_vv_test_loo_inter))

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(df_vv_train_drop_loo_inter_tmp)
df_vv_train_drop_loo_inter_minmax = pd.DataFrame(minMaxScaler.transform(df_vv_train_drop_loo_inter_tmp))
df_vv_test_drop_loo_inter_minmax = pd.DataFrame(minMaxScaler.transform(df_vv_test_drop_loo_inter))
df_vv_train_loo_inter_minmax['class'] = df_vv_train_loo_inter['class']
df_vv_train_drop_loo_inter_minmax['class'] = df_vv_train_drop_loo_inter['class']

colname = [x for x in df_vv_train_loo_inter.columns.drop('class')]
colname.append('class')
#colname.append('class')

df_vv_train_loo_inter_minmax.columns = colname
colname2 = [x for x in df_vv_train_drop_loo_inter.columns.drop('class')]
colname2.append('class')
df_vv_train_drop_loo_inter_minmax.columns = colname2

# df_vv_train_loo_inter_minmax.to_csv(save_path + "fin/df_vv_train_loo_inter_minmax.csv", encoding="EUC-KR", index=False)
# df_vv_test_loo_inter_minmax.to_csv(save_path + "fin/df_vv_test_loo_inter_minmax.csv", encoding="EUC-KR", index=False)
# df_vv_train_drop_loo_inter_minmax.to_csv(save_path + "fin/df_vv_train_drop_loo_inter_minmax.csv", encoding="EUC-KR", index=False)
# df_vv_test_drop_loo_inter_minmax.to_csv(save_path + "fin/df_vv_test_drop_loo_inter_minmax.csv", encoding="EUC-KR", index=False)
df_vv_train_loo_inter_minmax
df_vv_train_drop_loo_inter_minmax


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from pycaret.classification import *

###### data with all variables

clf2 = setup(data =df_vv_train_loo_inter_minmax, target = 'class', fix_imbalance=True,
             feature_selection=True, categorical_features=['at_age','vt_age'],
            numeric_features=['road_num','busstop_n','tree_n','subway_n'])

best_m2 = compare_models()
lr2 = create_model('lr')
tuned_lr2 = tune_model(lr2)
final_model2 = finalize_model(tuned_lr2)
evaluate_model(final_model2)

predictions = predict_model(final_model2, data = df_vv_train_loo_inter_minmax)
pred_label=predictions['Label']
pred_score=predictions['Score']
pred_label.value_counts()
label_test_y.value_counts()
print(accuracy_score(label_test_y , pred_label))

len(label_test_y )
len(pred_label)

predictions = predict_model(final_model2, data = df_vv_test_loo_inter_minmax)
df_vv_test_loo_inter_minmax.shape
df_vv_train_loo_inter_minmax.shape
df_vv_train_loo_inter_minmax
df_vv_test_loo_inter_minmax
df_vv_test_loo_inter_minmax.columns = df_vv_train_loo_inter_minmax.drop['class',axis=1].columns
df_vv_test_loo_inter_minmax.columns = df_vv_train_loo_inter_minmax.drop('class',axis=1).columns
df_vv_train_loo_inter_minmax
df_vv_test_loo_inter_minmax
predictions = predict_model(final_model2, data = df_vv_test_loo_inter_minmax)

pred_label=predictions['Label']
pred_score=predictions['Score']
pred_label.value_counts()
label_test_y.value_counts()
len(label_test_y )
len(pred_label)
print(accuracy_score(label_test_y , pred_label))

from sklearn.metrics import recall_score
print(recall_score(label_test_y , pred_label))
print(recall_score(label_test_y , pred_label, average=None))

fi0 = pd.DataFrame({'Feature': get_config('X_train').columns, 'Value' : final_model2.coef_[0]})
fi_2 = pd.concat([fi0,pd.DataFrame(final_model2.coef_[1]),pd.DataFrame(final_model2.coef_[2])],axis=1).sort_values(by='Value', key=abs,ascending=False)
fi_2.columns = ['feature','byc','pm','motor']
fi_2
fi_2.head(30)

clf3 = setup(data =df_vv_train_drop_loo_inter_minmax, target = 'class', fix_imbalance=True,
             feature_selection=True,
            numeric_features=['road_num','busstop_n','tree_n','subway_n'])

###### data without accident, age, gender, prot, mon-sat

best_m3 = compare_models()
lr3 = create_model('lr')
tuned_lr3 = tune_model(lr3)
final_model3 = finalize_model(tuned_lr3)
evaluate_model(final_model3)

df_vv_test_drop_loo_inter_minmax.columns = df_vv_train_drop_loo_inter_minmax.drop('class',axis=1).columns
predictions2 = predict_model(final_model3, data = df_vv_test_drop_loo_inter_minmax)

pred_label2=predictions2['Label']
pred_score2=predictions2['Score']
pred_label2.value_counts()
label_test_y.value_counts()
print(accuracy_score(label_test_y , pred_label2))
print(recall_score(label_test_y , pred_label2, average=None))

fi3 = pd.DataFrame({'Feature': get_config('X_train').columns, 'Value' : final_model3.coef_[0]})
fi_3 = pd.concat([fi0,pd.DataFrame(final_model3.coef_[1]),pd.DataFrame(final_model3.coef_[2])],axis=1).sort_values(by='Value', key=abs,ascending=False)
fi_3.columns = ['feature','byc','pm','motor']
fi_3
fi_3.head(30)

fi3 = pd.DataFrame({'Feature': get_config('X_train').columns, 'Value' : final_model3.coef_[0]})
fi_3 = pd.concat([fi0,pd.DataFrame(final_model3.coef_[1]),pd.DataFrame(final_model3.coef_[2])],axis=1).sort_values(by='Value', key=abs,ascending=False)
fi_3.columns = ['feature','byc','pm','motor']
fi_3

fi0 = pd.DataFrame({'Feature': get_config('X_train').columns, 'Value' : final_model3.coef_[0]})
fi1 = pd.DataFrame({'Feature': get_config('X_train').columns, 'Value' : final_model3.coef_[1]})
fi2 = pd.DataFrame({'Feature': get_config('X_train').columns, 'Value' : final_model3.coef_[2]})
fi_3 = pd.concat([fi0,fi1,fi2],axis=1)
fi_3.columns = ['feature','byc','d1','pm','d2','motor']
fi_3 = fi_3.drop(['d1','d2'],axis=1).sort_values(by='byc', key=abs,ascending=False)
#fi_3.columns = ['feature','byc','pm','motor']

fi_3
fi_3.head(30)

np.abs(fi_2['byc']) + np.abs(fi_2['pm']) + np.abs(fi_2['motor'])
fi_2['abs_sum'] = np.abs(fi_2['byc']) + np.abs(fi_2['pm']) + np.abs(fi_2['motor'])
fi_2.sort_values(by='abs_sum',ascending=False).head(30)

fi_3['abs_sum'] = np.abs(fi_3['byc']) + np.abs(fi_3['pm']) + np.abs(fi_3['motor'])
fi_3.sort_values(by='abs_sum',ascending=False)
fi_3['abs_sum'] = np.abs(fi_3['byc']) + np.abs(fi_3['pm']) + np.abs(fi_3['motor'])
fi_3.sort_values(by='abs_sum',ascending=False.head(30)

fi_3['abs_sum'] = np.abs(fi_3['byc']) + np.abs(fi_3['pm']) + np.abs(fi_3['motor'])
fi_3.sort_values(by='abs_sum',ascending=False).head(30)

fi_2.sort_values(by='abs_sum',ascending=False).to_csv(save_path + 'feature_importance_all.csv', encoding='EUC-KR')
fi_3.sort_values(by='abs_sum',ascending=False).to_csv(save_path + 'feature_importance_drop.csv', encoding='EUC-KR')
get_ipython().run_line_magic('save', 'modeling_with_fin_at_dataset.py 1-300')
