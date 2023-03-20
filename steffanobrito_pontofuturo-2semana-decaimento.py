

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier,LGBMRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_log_error,make_scorer  
from sklearn import preprocessing
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',900)
# from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory




train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
df= pd.concat([train, test])

df['Date'] = pd.to_datetime(df['Date'])

# Create date columns
le = preprocessing.LabelEncoder()
df['Day_num'] = le.fit_transform(df.Date)
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df.sample(5)
df['Province_State'].fillna('Vazio',inplace=True)
df['Local']=np.where(df['Province_State']== 'Vazio',df['Country_Region'],df['Country_Region']+'/'+df['Province_State'])
df.head()
df=df[df['Month']>2]
df.sample(5)
df_test=df[df['ForecastId']>0]
df_test.tail(2)


df['Date']=df['Date'].astype('str')
df=df[df['Id']>0]
df['ConfirmedCases'].fillna(0,inplace=True)
dft=df.pivot_table(index='Local',columns='Date',values='ConfirmedCases').reset_index()
dft.head()
df_s1=dft[['2020-03-25',
 '2020-03-26',
 '2020-03-27',
 '2020-03-28',
 '2020-03-29',
 '2020-03-30',
 '2020-03-31']]
df_s1['au1']=(df_s1['2020-03-31']/df_s1['2020-03-30'])-1
df_s1['au2']=(df_s1['2020-03-30']/df_s1['2020-03-29'])-1
df_s1['au3']=(df_s1['2020-03-29']/df_s1['2020-03-28'])-1
df_s1['au4']=(df_s1['2020-03-28']/df_s1['2020-03-27'])-1
df_s1['au5']=(df_s1['2020-03-27']/df_s1['2020-03-26'])-1
df_s1['au6']=(df_s1['2020-03-26']/df_s1['2020-03-25'])-1
#df_s1['au1']=(df_s1['2020-03-31']/df_s1['2020-03-30'])-1
dfs1=df_s1.filter(regex='au')
xs1=dfs1.quantile(axis=1)
dft['Mediana_cres_sem_1']=xs1

dfs1['d1']=(dfs1['au1']/dfs1['au2'])-1
dfs1['d2']=(dfs1['au2']/dfs1['au3'])-1
dfs1['d3']=(dfs1['au3']/dfs1['au4'])-1
dfs1['d4']=(dfs1['au4']/dfs1['au5'])-1
dfs1['d5']=(dfs1['au5']/dfs1['au6'])-1

s1=dfs1.filter(regex='d')
x_s1=s1.quantile(axis=1)
dft['Decaimento_semana']=(-1)*x_s1



df_s2=dft[['2020-03-18',
 '2020-03-19',
 '2020-03-20',
 '2020-03-21',
 '2020-03-22',
 '2020-03-23',
 '2020-03-24']]

df_s2['au1']=(df_s2['2020-03-24']/df_s2['2020-03-23'])-1
df_s2['au2']=(df_s2['2020-03-23']/df_s2['2020-03-22'])-1
df_s2['au3']=(df_s2['2020-03-22']/df_s2['2020-03-21'])-1
df_s2['au4']=(df_s2['2020-03-21']/df_s2['2020-03-20'])-1
df_s2['au5']=(df_s2['2020-03-20']/df_s2['2020-03-19'])-1
df_s2['au6']=(df_s2['2020-03-19']/df_s2['2020-03-18'])-1

dfs2=df_s2.filter(regex='au')
xs2=dfs2.quantile(axis=1)
dft['Mediana_cres_sem_2']=xs2

dfs2['d1']=(dfs2['au1']/dfs2['au2'])-1
dfs2['d2']=(dfs2['au2']/dfs2['au3'])-1
dfs2['d3']=(dfs2['au3']/dfs2['au4'])-1
dfs2['d4']=(dfs2['au4']/dfs2['au5'])-1
dfs2['d5']=(dfs2['au5']/dfs2['au6'])-1

s2=dfs2.filter(regex='d')
x_s2=s2.quantile(axis=1)
dft['Decaimento_semana_2']=(-1)*x_s2
dft['Cres1']=np.where(dft['2020-03-17']==0,1,(dft['2020-03-24']/(dft['2020-03-17'])))

dft['Cres1']=np.where(dft['2020-03-17']==0,1,(dft['2020-03-24']/(dft['2020-03-17'])))
#dft.head()
dft['Crescimento_1']=np.power((dft['Cres1']),1/7) - 1
#dft.head()


dft['Cres2']=np.where(dft['2020-03-24']==0,1,(dft['2020-03-31']/(dft['2020-03-24'])))
#dft.head()
dft['Crescimento_2']=np.power((dft['Cres2']),1/7) - 1
#dft.head()
dft.drop(columns=['Cres1','Cres2'],inplace=True)

dft['Decay']=np.power(dft['Crescimento_2']/dft['Crescimento_1'],1/7) - 1
Beta0=0.941
Beta1=-0.1692
dft['Cres_2020-04-01']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))
dft['Cres_2020-04-02']=dft['Cres_2020-04-01']*((dft['Cres_2020-04-01']*(Beta1)+Beta0))
dft['Cres_2020-04-03']=dft['Cres_2020-04-02']*((dft['Cres_2020-04-02']*(Beta1)+Beta0))
dft['Cres_2020-04-04']=dft['Cres_2020-04-03']*((dft['Cres_2020-04-03']*(Beta1)+Beta0))
dft['Cres_2020-04-05']=dft['Cres_2020-04-04']*((dft['Cres_2020-04-04']*(Beta1)+Beta0))
dft['Cres_2020-04-06']=dft['Cres_2020-04-05']*((dft['Cres_2020-04-05']*(Beta1)+Beta0))
dft['Cres_2020-04-07']=dft['Cres_2020-04-06']*((dft['Cres_2020-04-06']*(Beta1)+Beta0))
dft['Cres_2020-04-08']=dft['Cres_2020-04-07']*((dft['Cres_2020-04-07']*(Beta1)+Beta0))
dft['Cres_2020-04-09']=dft['Cres_2020-04-08']*((dft['Cres_2020-04-08']*(Beta1)+Beta0))
dft['Cres_2020-04-10']=dft['Cres_2020-04-09']*((dft['Cres_2020-04-09']*(Beta1)+Beta0))
dft['Cres_2020-04-11']=dft['Cres_2020-04-10']*((dft['Cres_2020-04-10']*(Beta1)+Beta0))
dft['Cres_2020-04-12']=dft['Cres_2020-04-11']*((dft['Cres_2020-04-11']*(Beta1)+Beta0))
dft['Cres_2020-04-13']=dft['Cres_2020-04-12']*((dft['Cres_2020-04-12']*(Beta1)+Beta0))
dft['Cres_2020-04-14']=dft['Cres_2020-04-13']*((dft['Cres_2020-04-13']*(Beta1)+Beta0))
dft['Cres_2020-04-15']=dft['Cres_2020-04-14']*((dft['Cres_2020-04-14']*(Beta1)+Beta0))
dft['Cres_2020-04-16']=dft['Cres_2020-04-15']*((dft['Cres_2020-04-15']*(Beta1)+Beta0))
dft['Cres_2020-04-17']=dft['Cres_2020-04-16']*((dft['Cres_2020-04-16']*(Beta1)+Beta0))
dft['Cres_2020-04-18']=dft['Cres_2020-04-17']*((dft['Cres_2020-04-17']*(Beta1)+Beta0))
dft['Cres_2020-04-19']=dft['Cres_2020-04-18']*((dft['Cres_2020-04-18']*(Beta1)+Beta0))
dft['Cres_2020-04-20']=dft['Cres_2020-04-19']*((dft['Cres_2020-04-19']*(Beta1)+Beta0))
dft['Cres_2020-04-21']=dft['Cres_2020-04-20']*((dft['Cres_2020-04-20']*(Beta1)+Beta0))
dft['Cres_2020-04-22']=dft['Cres_2020-04-21']*((dft['Cres_2020-04-21']*(Beta1)+Beta0))
dft['Cres_2020-04-23']=dft['Cres_2020-04-22']*((dft['Cres_2020-04-22']*(Beta1)+Beta0))
dft['Cres_2020-04-24']=dft['Cres_2020-04-23']*((dft['Cres_2020-04-23']*(Beta1)+Beta0))
dft['Cres_2020-04-25']=dft['Cres_2020-04-24']*((dft['Cres_2020-04-24']*(Beta1)+Beta0))
dft['Cres_2020-04-26']=dft['Cres_2020-04-25']*((dft['Cres_2020-04-25']*(Beta1)+Beta0))
dft['Cres_2020-04-27']=dft['Cres_2020-04-26']*((dft['Cres_2020-04-26']*(Beta1)+Beta0))
dft['Cres_2020-04-28']=dft['Cres_2020-04-27']*((dft['Cres_2020-04-27']*(Beta1)+Beta0))
dft['Cres_2020-04-29']=dft['Cres_2020-04-28']*((dft['Cres_2020-04-28']*(Beta1)+Beta0))
dft['Cres_2020-04-30']=dft['Cres_2020-04-29']*((dft['Cres_2020-04-29']*(Beta1)+Beta0))






dft['2020-04-01']=(1+dft['Cres_2020-04-01'])*dft['2020-03-31']
dft['2020-04-02']=(1+dft['Cres_2020-04-02'])*dft['2020-04-01']
dft['2020-04-03']=(1+dft['Cres_2020-04-03'])*dft['2020-04-02']
dft['2020-04-04']=(1+dft['Cres_2020-04-04'])*dft['2020-04-03']
dft['2020-04-05']=(1+dft['Cres_2020-04-05'])*dft['2020-04-04']
dft['2020-04-06']=(1+dft['Cres_2020-04-06'])*dft['2020-04-05']
dft['2020-04-07']=(1+dft['Cres_2020-04-07'])*dft['2020-04-06']
dft['2020-04-08']=(1+dft['Cres_2020-04-08'])*dft['2020-04-07']
dft['2020-04-09']=(1+dft['Cres_2020-04-09'])*dft['2020-04-08']
dft['2020-04-10']=(1+dft['Cres_2020-04-10'])*dft['2020-04-09']
dft['2020-04-11']=(1+dft['Cres_2020-04-11'])*dft['2020-04-10']
dft['2020-04-12']=(1+dft['Cres_2020-04-12'])*dft['2020-04-11']
dft['2020-04-13']=(1+dft['Cres_2020-04-13'])*dft['2020-04-12']
dft['2020-04-14']=(1+dft['Cres_2020-04-14'])*dft['2020-04-13']
dft['2020-04-15']=(1+dft['Cres_2020-04-15'])*dft['2020-04-14']
dft['2020-04-16']=(1+dft['Cres_2020-04-16'])*dft['2020-04-15']
dft['2020-04-17']=(1+dft['Cres_2020-04-17'])*dft['2020-04-16']
dft['2020-04-18']=(1+dft['Cres_2020-04-18'])*dft['2020-04-17']
dft['2020-04-19']=(1+dft['Cres_2020-04-19'])*dft['2020-04-18']
dft['2020-04-20']=(1+dft['Cres_2020-04-20'])*dft['2020-04-19']
dft['2020-04-21']=(1+dft['Cres_2020-04-21'])*dft['2020-04-20']
dft['2020-04-22']=(1+dft['Cres_2020-04-22'])*dft['2020-04-21']
dft['2020-04-23']=(1+dft['Cres_2020-04-23'])*dft['2020-04-22']
dft['2020-04-24']=(1+dft['Cres_2020-04-24'])*dft['2020-04-23']
dft['2020-04-25']=(1+dft['Cres_2020-04-25'])*dft['2020-04-24']
dft['2020-04-26']=(1+dft['Cres_2020-04-26'])*dft['2020-04-25']
dft['2020-04-27']=(1+dft['Cres_2020-04-27'])*dft['2020-04-26']
dft['2020-04-28']=(1+dft['Cres_2020-04-28'])*dft['2020-04-27']
dft['2020-04-29']=(1+dft['Cres_2020-04-29'])*dft['2020-04-28']
dft['2020-04-30']=(1+dft['Cres_2020-04-30'])*dft['2020-04-29']


dfm=df.pivot_table(index='Local',columns='Date',values='Fatalities').reset_index()

dft['mortes']=dfm['2020-03-31']/dft['2020-03-31']
mortes_adj=dfm['2020-03-31'].sum(axis=0) / dft['2020-03-31'].sum(axis=0) 
mortes_adj
#dft.loc(dft['mortes']>(2*mortes_adj),'mortes')=(2*mortes_adj)
#dft.loc(dft['mortes']<(mortes_adj/2),'mortes')=(mortes_adj/2)
dft['mortes']=np.where(dft['mortes']>(2*mortes_adj),(2*mortes_adj),np.where(dft['mortes']<(mortes_adj/2),(mortes_adj/2),dft['mortes']))
dfi=dft[['Local', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04',
       '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09',
       '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13', '2020-03-14',
       '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19',
       '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24',
       '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29',
       '2020-03-30', '2020-03-31',  '2020-04-01', '2020-04-02', '2020-04-03',
       '2020-04-04', '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',
       '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13',
       '2020-04-14', '2020-04-15', '2020-04-16', '2020-04-17', '2020-04-18',
       '2020-04-19', '2020-04-20', '2020-04-21', '2020-04-22', '2020-04-23',
       '2020-04-24', '2020-04-25', '2020-04-26', '2020-04-27', '2020-04-28',
       '2020-04-29', '2020-04-30']]
dfi.head()
df = dfi.melt('Local', var_name='Date', value_name='ConfirmedCases')
df.head()

df=pd.merge(df,dft[['Local','mortes']],on='Local',how='left')
df['Fatalities']=df['ConfirmedCases']*df['mortes']
df[df['Local']=='Brazil']


df_test.head()
df_test=df_test.drop(columns=['ConfirmedCases','Fatalities'])
df_test['Date']=df_test['Date'].astype('str')
df_test=pd.merge(df_test,df[['Local','Date','ConfirmedCases','Fatalities']],on=['Local','Date'],how='left')
submission=df_test[['ForecastId','ConfirmedCases','Fatalities']]
submission.to_csv('submission.csv',index=None)
submission.sample(10)


