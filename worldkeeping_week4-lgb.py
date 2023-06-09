# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## importing packages

import lightgbm as lgb

import numpy as np

import pandas as pd



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, mean_absolute_error,roc_auc_score

from google.cloud import bigquery

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import Lasso,Ridge

from datetime import date

from datetime import timedelta

import gc

import warnings

warnings.filterwarnings("ignore")

def MyLabelEncodeSingle(col):

    levels=col.unique().tolist()

    for l in levels:

        if l is np.nan:

            levels.remove(np.nan)

    levelmap={e:i for i,e in enumerate(levels)}

    return col.map(levelmap)
## defining constants

VAL_DAYS = 7

MAD_FACTOR = 0.5

DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000]



SEED = 1990

## reading data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

#https://www.kaggle.com/rohanrao/covid19-forecasting-metadata

region_metadata = pd.read_csv('/kaggle/input/covid19-forecasting-metadata/region_metadata.csv')

region_date_metadata = pd.read_csv('/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv')

## preparing data

train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on = ["Province_State", "Country_Region", "Date"], how = "left")

test = test[~test.Date.isin(train.Date.unique())]



df_panel = pd.concat([train, test], sort = False)



# combining state and country into 'geography'

#df_panel["geography"] = df_panel.Country_Region.astype(str) + ": " + df_panel.Province_State.astype(str)

#df_panel.loc[df_panel.Province_State.isna(), "geography"] = df_panel[df_panel.Province_State.isna()].Country_Region

df_panel["Country_Region"]=df_panel["Country_Region"].fillna('Unknown')

df_panel["Province_State"]=df_panel["Province_State"].fillna('Unknown')

# fixing data issues with cummax

#df_panel.ConfirmedCases = df_panel.groupby("geography")["ConfirmedCases"].cummax()

#df_panel.Fatalities = df_panel.groupby("geography")["Fatalities"].cummax()

df_panel.ConfirmedCases = df_panel.groupby(["Country_Region","Province_State"])["ConfirmedCases"].cummax()

df_panel.Fatalities = df_panel.groupby(["Country_Region","Province_State"])["Fatalities"].cummax()

# merging external metadata

region_metadata["Country_Region"]=region_metadata["Country_Region"].fillna('Unknown')

region_metadata["Province_State"]=region_metadata["Province_State"].fillna('Unknown')

region_date_metadata["Country_Region"]=region_date_metadata["Country_Region"].fillna('Unknown')

region_date_metadata["Province_State"]=region_date_metadata["Province_State"].fillna('Unknown')

df_panel = df_panel.merge(region_metadata, on = ["Country_Region", "Province_State"], how = "left")

df_panel = df_panel.merge(region_date_metadata, on = ["Country_Region", "Province_State", "Date"], how = "left")



# label encoding continent

df_panel.continent = MyLabelEncodeSingle(df_panel.continent)

df_panel.Date = pd.to_datetime(df_panel.Date, format = "%Y-%m-%d")



#df_panel.sort_values(["geography", "Date"], inplace = True)



## feature engineering

min_date_train = np.min(df_panel[~df_panel.Id.isna()].Date)

max_date_train = np.max(df_panel[~df_panel.Id.isna()].Date)



min_date_test = np.min(df_panel[~df_panel.ForecastId.isna()].Date)

max_date_test = np.max(df_panel[~df_panel.ForecastId.isna()].Date)



n_dates_test = len(df_panel[~df_panel.ForecastId.isna()].Date.unique())



print("Train date range:", str(min_date_train), " - ", str(max_date_train))

print("Test date range:", str(min_date_test), " - ", str(max_date_test))



# creating lag features

for lag in range(1, 41):

    df_panel[f"lag_{lag}_cc"] = df_panel.groupby(["Country_Region","Province_State"])["ConfirmedCases"].shift(lag)

    df_panel[f"lag_{lag}_ft"] = df_panel.groupby(["Country_Region","Province_State"])["Fatalities"].shift(lag)

    df_panel[f"lag_{lag}_rc"] = df_panel.groupby(["Country_Region","Province_State"])["Recoveries"].shift(lag)



for case in DAYS_SINCE_CASES:

    df_panel = df_panel.merge(df_panel[df_panel.ConfirmedCases >= case].groupby(["Country_Region","Province_State"])["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = ["Country_Region","Province_State"], how = "left")

#https://www.kaggle.com/jasonbenner/world-bank-datasets#World_Happiness_Index.csv

world_happiness = pd.read_csv("../input/world-bank-datasets/World_Happiness_Index.csv")

world_happiness=world_happiness.iloc[:,:19]

world_happiness.columns=[c.replace('(','').replace(')','').replace('(','').replace(',','').replace('-','_').replace('/','_').replace(' ','_') 

                               for c in world_happiness.columns]
average_year={}

temp_matrix=world_happiness.iloc[:,2:]

for y in world_happiness.Year.unique():

    average_year[y]=temp_matrix.loc[world_happiness.Year==y,:].mean()

del temp_matrix

gc.collect()
distance=0

while world_happiness.isna().sum().sum()!=0:

    for y in world_happiness.Year.unique():

        yhat=y-distance

        if yhat>2018:

            yhat=2018

        elif yhat<2005:

            yhat=2005

        for c in world_happiness.columns[2:]:

            world_happiness.loc[world_happiness.Year==y,c]=world_happiness.loc[world_happiness.Year==y,c].fillna(average_year[yhat][c])

        yhat=y+distance

        if yhat>2018:

            yhat=2018

        elif yhat<2005:

            yhat=2005

        for c in world_happiness.columns[2:]:

            world_happiness.loc[world_happiness.Year==y,c]=world_happiness.loc[world_happiness.Year==y,c].fillna(average_year[yhat][c])

        distance += 1
world_happiness_latest = world_happiness.groupby('Country_name').nth(-1)

world_happiness_first = world_happiness.groupby('Country_name').agg('first')

world_happiness_last = world_happiness.groupby('Country_name').agg('last')

world_happiness_count = world_happiness.groupby('Country_name').count()

world_happiness_range=(world_happiness_last-world_happiness_first)/world_happiness_count

world_happiness_range.drop("Year", axis=1, inplace=True)

world_happiness_latest.drop("Year", axis=1, inplace=True)

world_happiness_range.columns=[c+'_range' for c in world_happiness_range.columns]

world_happiness_latest.columns=[c+'_latest' for c in world_happiness_latest.columns]

world_happiness_grouped=pd.concat((world_happiness_latest,world_happiness_range),axis=1).reset_index()
malaria_world_health = pd.read_csv("../input/world-bank-datasets/Malaria_World_Health_Organization.csv")

malaria_world_health.columns=[c.replace(' ','_') for c in malaria_world_health.columns]
human_development = pd.read_csv("../input/world-bank-datasets/Human_Development_Index.csv")

human_development.columns=[c.replace(')','').replace('(','').replace(' ','_') for c in human_development.columns]

human_development['Gross_national_income_GNI_per_capita_2018']= human_development['Gross_national_income_GNI_per_capita_2018'].apply(lambda x: x if x!=x else x.replace(',','')).astype(float)
#https://www.kaggle.com/nightranger77/covid19-demographic-predictors

night_ranger = pd.read_csv("../input/covid19-demographic-predictors/covid19_by_country.csv")

night_ranger.columns=[c.replace(' ','_') for c in night_ranger.columns]

night_ranger = night_ranger[night_ranger.Country != "Georgia"]

night_ranger=night_ranger[['Country','Median_Age','GDP_2018','Crime_Index','Population_2020','Smoking_2016','Females_2018']]
#https://www.kaggle.com/londeen/world-happiness-report-2020

happiness_df = pd.read_csv("../input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv")

happiness_df.columns=[c.replace(':','').replace('+','').replace(' ','_') for c in happiness_df.columns]

happiness_df['Regional_indicator']=MyLabelEncodeSingle(happiness_df['Regional_indicator'])
#https://www.kaggle.com/alizahidraja/world-population-by-age-group-2020

age_df = pd.read_csv("../input/world-population-by-age-group-2020/WorldPopulationByAge2020.csv")

age_df['AgeGrp']=MyLabelEncodeSingle(age_df['AgeGrp'])

def processAge(df):

    ageindex=df['AgeGrp']

    sexsum=df[['PopMale', 'PopFemale', 'PopTotal']].sum()

    mp=sexsum['PopMale']/sexsum['PopTotal']

    fp=sexsum['PopFemale']/sexsum['PopTotal']

    p0=df.loc[ageindex==0,'PopTotal'].values[0]/sexsum['PopTotal']

    p1=df.loc[ageindex==1,'PopTotal'].values[0]/sexsum['PopTotal']

    p2=df.loc[ageindex==2,'PopTotal'].values[0]/sexsum['PopTotal']

    p3=df.loc[ageindex==3,'PopTotal'].values[0]/sexsum['PopTotal']

    m0=df.loc[ageindex==0,'PopMale'].values[0]/sexsum['PopMale']

    m1=df.loc[ageindex==1,'PopMale'].values[0]/sexsum['PopMale']

    m2=df.loc[ageindex==2,'PopMale'].values[0]/sexsum['PopMale']

    m3=df.loc[ageindex==3,'PopMale'].values[0]/sexsum['PopMale']

    f0=df.loc[ageindex==0,'PopFemale'].values[0]/sexsum['PopFemale']

    f1=df.loc[ageindex==1,'PopFemale'].values[0]/sexsum['PopFemale']

    f2=df.loc[ageindex==2,'PopFemale'].values[0]/sexsum['PopFemale']

    f3=df.loc[ageindex==3,'PopFemale'].values[0]/sexsum['PopFemale']

    return pd.DataFrame({'MaleP':mp,'MaleP_0':m0,'MaleP_1':m1,'MaleP_2':m2,'MaleP_3':m3,'FemaleP':fp,

                         'FemaleP_0':f0,'FemaleP_1':f1,'FemaleP_2':f2,'FemaleP_3':f3,'PopTotal':sexsum['PopTotal'],

                         'Pop_0':p0,'Pop_1':p1,'Pop_2':p2,'Pop_3':p3},index=[0])

age_df=age_df.groupby('Location').apply(processAge).reset_index().drop('level_1',axis=1)
#https://www.kaggle.com/tanuprabhu/population-by-country-2020

pop_df = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")

pop_df.columns=[c.replace('.',' ').split(' ')[0]+'_pop2020' for c in pop_df.columns]

percent_col=['Yearly_pop2020','Urban_pop2020', 'World_pop2020']

def depercent(x):

    if x=='N.A.':

        return np.nan 

    else:

        return float(x.replace('%',''))

for c in percent_col:

    pop_df[c]=pop_df[c].apply(lambda x: depercent(x))

pop_df=pop_df.replace('N.A.',np.nan)

pop_df[['Population_pop2020', 'Yearly_pop2020',

       'Net_pop2020', 'Density_pop2020', 'Land_pop2020', 'Migrants_pop2020',

       'Fert_pop2020', 'Med_pop2020', 'Urban_pop2020', 'World_pop2020']]=pop_df[['Population_pop2020', 'Yearly_pop2020',

       'Net_pop2020', 'Density_pop2020', 'Land_pop2020', 'Migrants_pop2020',

       'Fert_pop2020', 'Med_pop2020', 'Urban_pop2020', 'World_pop2020']].astype(float)
#https://www.kaggle.com/hbfree/covid19formattedweatherjan22march24

weather_df = pd.read_csv("../input/covid19formattedweatherjan22march24/covid_dataset.csv")

weather_df=weather_df[['Province/State',

'Country/Region',

'lat',

'long',

'day',

'pop',

'urbanpop',

'density',

'medianage',

'smokers',

'health_exp_pc',

'hospibed',

'temperature',

'humidity']]
weather_df=weather_df.replace(-999,np.nan)

weather_df['Province/State']=weather_df['Province/State'].fillna('Unknown')

weather_df['day']=pd.to_datetime('2020-01-22')+weather_df['day'].apply(lambda x: timedelta(days=x))

weather_df['month']=weather_df['day'].dt.month

weather_df.drop('day',axis=1,inplace=True)

weather_df=weather_df.groupby(['Country/Region','Province/State','month']).mean().reset_index()

weather_df_latest = weather_df.groupby(['Country/Region','Province/State']).nth(-1).reset_index()

weather_df_latest['month']=4

weather_df=pd.concat((weather_df,weather_df_latest),sort=True,axis=0,ignore_index=True)
#https://www.kaggle.com/danevans/world-bank-wdi-212-health-systems

healthsys_df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")

healthsys_df.columns=[c.replace('-','_') for c in healthsys_df.columns]

healthsys_df.drop('World_Bank_Name',axis=1,inplace=True)

nan_country=healthsys_df[['Country_Region', 'Province_State']].isna().all(axis=1)

healthsys_df=healthsys_df.loc[nan_country==False,:].reset_index(drop=True)

healthsys_df['Province_State']=healthsys_df['Province_State'].fillna('Unknown')
#https://www.kaggle.com/koryto/countryinfo

compre_df = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

compre_df['region']=compre_df['region'].fillna('Unknown')

keepcol=['region', 'country', 'tests',

       'testpop', 'density', 'medianage', 'urbanpop', 'quarantine', 'schools',

       'publicplace', 'gatheringlimit', 'gathering', 'nonessential',

       'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64',

       'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019',

       'healthexp', 'healthperpop', 'fertility', 'firstcase']
def tempfun(x):

    if x is np.nan:

        return x

    else:

        return float(x.replace(',',''))

for c in ['gdp2019','healthexp']:

    compre_df[c]=compre_df[c].apply(lambda x: tempfun(x) )

    todate_col=['quarantine', 'schools','publicplace', 'gathering', 'nonessential','firstcase']

for c in todate_col:

    compre_df[c]= (pd.to_datetime(date.today())-pd.to_datetime(compre_df[c])).dt.days.astype(float)

compre_df=compre_df[keepcol]
#https://www.kaggle.com/imdevskp/sars-outbreak-2003-complete-dataset

sars_df = pd.read_csv("../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")
def getProvince(x):

    x_seg=x.split(',')

    if len(x_seg)==2:

        if 'SAR' in x_seg[0]:

            return x_seg[0][:-4]

        else:

            return x_seg[0]

    else:

        return np.nan

sars_df['Province']=sars_df['Country'].apply(lambda x: getProvince(x))

sars_df['Country']=sars_df['Country'].apply(lambda x: x.split(',')[-1])

sars_df['Country']=sars_df['Country'].replace('Viet Nam','Vietnam')

def getSlope(ses,segs):

    segsize=np.floor(len(ses)/segs)

    slope=[]

    for i in range(segs):

        if i==segs-1:

            slope.append((ses[-1]-ses[int(i*segsize)])/(len(ses)-1-i*segsize))

        else:

            slope.append((ses[int((i+1)*segsize-1)]-ses[int(i*segsize)])/(segsize-1))

    return slope   

def aggSARS(df):

    df=df.sort_values('Date')

    case=df['Cumulative number of case(s)']

    death=df['Number of deaths'].cumsum()

    recover=df['Number recovered'].cumsum()

    Sars_dict={}

    Sars_dict['SARS_CaseMax']=case.max()

    Sars_dict['SARS_DeathMax']=death.max()

    Sars_dict['SARS_RecoverMax']=recover.max()

    segs=df['Date'].apply(lambda x: x.split('-')[1]).nunique()

    for i,s in enumerate(getSlope(case.values,segs)):

        Sars_dict['SARS_Case_'+str(i)]=s

    for i,s in enumerate(getSlope(death.values,segs)):

        Sars_dict['SARS_Death_'+str(i)]=s

    for i,s in enumerate(getSlope(recover.values,segs)):

        Sars_dict['SARS_Recover_'+str(i)]=s

    return pd.DataFrame(Sars_dict,index=[0])

sars_df['Province']=sars_df['Province'].fillna('Unknown')

sars_df_grouped=sars_df.groupby(['Country','Province']).apply(aggSARS).reset_index().drop('level_2',axis=1)

sars_df_grouped=sars_df_grouped.rename(columns={'Country':'Country_Region','Province':'Province_State'})
#https://www.kaggle.com/worldkeeping/2009-h1n1-flu-with-readable-filename

h1n1_df = pd.read_csv("../input/2009-h1n1-flu-with-readable-filename/Pandemic_H1N1_2009.csv")
spelreplace={'\xa0\xa0\xa0\xa0\xa0French Polynesia, FOC':'French Polynesia',

       '\xa0\xa0\xa0\xa0\xa0Guadaloupe, FOC':'Guadaloupe',

       '\xa0\xa0\xa0\xa0\xa0Martinique, FOC':'Martinique',

       '\xa0\xa0\xa0\xa0\xa0New Caledonia, FOC':'New Caledonia',

       '\xa0\xa0\xa0\xa0\xa0Saint Martin, FOC':'Saint Martin', 

    '\xa0\xa0\xa0\xa0\xa0Netherlands, Aruba':'Netherlands, Aruba',

       '\xa0\xa0\xa0\xa0\xa0Netherlands Antilles, Curaçao':'Netherlands Antilles, Curaçao',

       '\xa0\xa0\xa0\xa0\xa0Netherlands Antilles, Sint Maarten':'Netherlands Antilles, Sint Maarten',

     '\xa0\xa0\xa0\xa0French Polynesia, FOC':'French Polynesia',

       '\xa0\xa0\xa0\xa0Martinique, FOC':'Martinique',

       '\xa0\xa0\xa0\xa0France, New Caledonia, FOC':'France, New Caledonia', 

       '\xa0\xa0\xa0\xa0\xa0Netherlands Antilles, Curaçao *':'Netherlands Antilles, Curaçao', 

       '\xa0\xa0\xa0\xa0\xa0Netherlands Antilles, Curaçao **':'Netherlands Antilles, Curaçao',

            '\xa0\xa0\xa0\xa0\xa0Guernsey, Crown Dependency':'Guernsey, Crown Dependency',

       '\xa0\xa0\xa0\xa0\xa0Isle of Man, Crown Dependency':'Isle of Man, Crown Dependency',

       '\xa0\xa0\xa0\xa0\xa0Jersey, Crown Dependency':'Jersey, Crown Dependency',

        '\xa0\xa0\xa0\xa0\xa0Puerto Rico':'Puerto Rico','*':'','Guatemala\xa0':'Guatemala',

            'Korea, Republic of':'Korea, South','\xa0\xa0\xa0\xa0\xa0Virgin Islands':'Virgin Islands',

            'Costa Rica*':'Costa Rica','Morocco *':'Morocco','Viet Nam':'Vietnam','United States of America*':'United States of America'}
h1n1_df['Country']=h1n1_df['Country'].replace(spelreplace)
def getP(x):

    x_seg=x.split(',')

    if len(x_seg)>1:

        if x_seg[0].strip() != 'Korea':

            if x_seg[1].strip() == 'UKOT':

                return x_seg[0]

            elif x_seg[-1].strip() == 'Crown Dependency':

                return x_seg[-2]

            else:   

                return x_seg[1]

    else:

        return np.nan
def getC(x):

    x_seg=x.split(',')

    if len(x_seg)>1:

        if x_seg[0].strip() != 'Korea':

            if x_seg[1].strip() == 'UKOT':

                return 'United Kingdom'

            elif x_seg[-1].strip() == 'Crown Dependency':

                return 'United Kingdom'

            else:   

                return x_seg[0]

    elif x == 'United States of America':

        return 'US'

    else:

        return x
h1n1_df['Province']=h1n1_df['Country'].apply(getP)

h1n1_df['Country']=h1n1_df['Country'].apply(getC)
def getSlope(ses,segs):

    segsize=np.floor(len(ses)/segs)

    slope=[]

    for i in range(segs):

        if i==segs-1:

            slope.append((ses[-1]-ses[int(i*segsize)])/(len(ses)-1-i*segsize))

        else:

            slope.append((ses[int((i+1)*segsize-1)]-ses[int(i*segsize)])/(segsize-1))

    return slope   

def aggh1n1(df):

    df=df.sort_values('Update Time')

    case=df['Cases']

    death=df['Deaths']

    Sars_dict={}

    Sars_dict['H1N1_CaseMax']=case.max()

    Sars_dict['H1N1_DeathMax']=death.max()

    segs=df['Update Time'].apply(lambda x: x.split('/')[0]).nunique()

    for i,s in enumerate(getSlope(case.values,segs)):

        Sars_dict['H1N1_Case_'+str(i)]=s

    for i,s in enumerate(getSlope(death.values,segs)):

        Sars_dict['H1N1_Death_'+str(i)]=s

    return pd.DataFrame(Sars_dict,index=[0])

h1n1_df['Province']=h1n1_df['Province'].fillna('Unknown')

h1n1_df_grouped=h1n1_df.groupby(['Country','Province']).apply(aggh1n1).reset_index().drop('level_2',axis=1)

h1n1_df_grouped=h1n1_df_grouped.rename(columns={'Country':'Country_Region','Province':'Province_State'})
def merge2layer(left,right):

    rightdf=right[['Country_Region','Province_State']]

    rightdf['mark']=1

    leftdf=left[['Country_Region','Province_State']]

    countryNoP=pd.merge(left=leftdf,right=rightdf,on=['Country_Region','Province_State'],how='left')

    firstdf=pd.merge(left=left,right=right,on=['Country_Region','Province_State'],how='left')

    nextidx=countryNoP['mark'].isna()

    firstdf=firstdf.loc[~nextidx,:]

    countryNoP_right=right.loc[right['Province_State'].isna(),:].drop('Province_State',axis=1)

    countryNoP_left=left.loc[nextidx,:]

    seconddf=pd.merge(left=countryNoP_left,right=countryNoP_right,on='Country_Region',how='left')

    finaldf=pd.concat((firstdf,seconddf),sort=True,axis=0,ignore_index=True)

    return finaldf
#all extra features calculated above

def extrafeatures(df):

    #print('before: {}'.format(len(df)))

    has_col=df.columns.tolist()

    df['UpToNow']=(pd.to_datetime(date.today())-pd.to_datetime(df['Date'])).dt.days.astype(float)

   # print('after UpToNow: {}'.format(len(df)))

    df = pd.merge(left=df, right=world_happiness_grouped, how='left', left_on='Country_Region', right_on='Country_name')

    df.drop("Country_name", axis=1, inplace=True)

    #print('after world_happiness: {}'.format(len(df)))

    df = pd.merge(left=df, right=malaria_world_health, how='left', left_on='Country_Region', right_on='Country')

    df.drop("Country", axis=1, inplace=True)

   # print('after malaria: {}'.format(len(df)))

    df = pd.merge(left=df, right=human_development, how='left', left_on='Country_Region', right_on='Country')

    df.drop("Country", axis=1, inplace=True)

   # print('after human: {}'.format(len(df)))

    df = pd.merge(left=df, right=night_ranger, how='left', left_on='Country_Region', right_on='Country')

    df.drop("Country", axis=1, inplace=True)

    #print('after night: {}'.format(len(df)))

    df = pd.merge(left=df, right=happiness_df, how='left', left_on='Country_Region', right_on='Country_name')

    df.drop('Country_name', axis=1, inplace=True)

    #print('after happiness: {}'.format(len(df)))

    df = pd.merge(left=df, right=age_df, how='left', left_on='Country_Region', right_on='Location')

    df.drop('Location', axis=1, inplace=True)

    #print('after age: {}'.format(len(df)))

    df = pd.merge(left=df, right=pop_df, how='left', left_on='Country_Region', right_on='Country_pop2020')

    df.drop('Country_pop2020', axis=1, inplace=True)

    #print('after pop: {}'.format(len(df)))

    df['month']=df['Date'].dt.month

    df = pd.merge(left=df, right=weather_df, how='left', left_on=['Country_Region','Province_State','month'], right_on=['Country/Region','Province/State','month'])

    df.drop(['Country/Region','Province/State','month'], axis=1, inplace=True)

    #print('after weather: {}'.format(len(df)))

    df = merge2layer(df,healthsys_df)

    #print('after healthsys: {}'.format(len(df)))

    df = pd.merge(left=df, right=compre_df, how='left', left_on=['Country_Region','Province_State'], right_on=['country','region'])

    df.drop(['country','region'], axis=1, inplace=True)

    #print('after compre: {}'.format(len(df)))

    df = merge2layer(df,sars_df_grouped)

    #print('after sars: {}'.format(len(df)))

    df = merge2layer(df,h1n1_df_grouped)

    #print('after h1n1: {}'.format(len(df)))

    #df['Country_Region']=MyLabelEncodeSingle(df['Country_Region'])

    #df['Province_State']=MyLabelEncodeSingle(df['Province_State'])

    df['thishour'] = df['Date'].dt.hour

    df['thisdayofweek'] = df['Date'].dt.dayofweek

    df['thisquarter'] = df['Date'].dt.quarter

    df['thismonth'] = df['Date'].dt.month

    df['thisdayofyear'] = df['Date'].dt.dayofyear

    df['thisdayofmonth'] = df['Date'].dt.day

    df['thisweekofyear'] = df['Date'].dt.weekofyear

    extra_col=[c for c in df.columns.tolist() if c not in has_col]

    return df,extra_col
## function for preparing features

def prepare_features(df, gap):

    

    df["perc_1_ac"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap}_ft"] - df[f"lag_{gap}_rc"]) / df[f"lag_{gap}_cc"]

    df["perc_1_cc"] = df[f"lag_{gap}_cc"] / df.population

    

    df["diff_1_cc"] = df[f"lag_{gap}_cc"] - df[f"lag_{gap + 1}_cc"]

    df["diff_2_cc"] = df[f"lag_{gap + 1}_cc"] - df[f"lag_{gap + 2}_cc"]

    df["diff_3_cc"] = df[f"lag_{gap + 2}_cc"] - df[f"lag_{gap + 3}_cc"]

    

    df["diff_1_ft"] = df[f"lag_{gap}_ft"] - df[f"lag_{gap + 1}_ft"]

    df["diff_2_ft"] = df[f"lag_{gap + 1}_ft"] - df[f"lag_{gap + 2}_ft"]

    df["diff_3_ft"] = df[f"lag_{gap + 2}_ft"] - df[f"lag_{gap + 3}_ft"]

    

    df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3

    df["diff_123_ft"] = (df[f"lag_{gap}_ft"] - df[f"lag_{gap + 3}_ft"]) / 3



    df["diff_change_1_cc"] = df.diff_1_cc / df.diff_2_cc

    df["diff_change_2_cc"] = df.diff_2_cc / df.diff_3_cc

    

    df["diff_change_1_ft"] = df.diff_1_ft / df.diff_2_ft

    df["diff_change_2_ft"] = df.diff_2_ft / df.diff_3_ft



    df["diff_change_12_cc"] = (df.diff_change_1_cc + df.diff_change_2_cc) / 2

    df["diff_change_12_ft"] = (df.diff_change_1_ft + df.diff_change_2_ft) / 2

    

    df["change_1_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 1}_cc"]

    df["change_2_cc"] = df[f"lag_{gap + 1}_cc"] / df[f"lag_{gap + 2}_cc"]

    df["change_3_cc"] = df[f"lag_{gap + 2}_cc"] / df[f"lag_{gap + 3}_cc"]



    df["change_1_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 1}_ft"]

    df["change_2_ft"] = df[f"lag_{gap + 1}_ft"] / df[f"lag_{gap + 2}_ft"]

    df["change_3_ft"] = df[f"lag_{gap + 2}_ft"] / df[f"lag_{gap + 3}_ft"]



    df["change_123_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"]

    df["change_123_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 3}_ft"]

    

    for case in DAYS_SINCE_CASES:

        df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")

        df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan



    df["country_flag"] = df.Province_State.isna().astype(int)

    df["density"] = df.population / df.area

    

    # target variable is log of change from last known value

    df["target_cc"] = np.log1p(df.ConfirmedCases) - np.log1p(df[f"lag_{gap}_cc"])

    df["target_ft"] = np.log1p(df.Fatalities) - np.log1p(df[f"lag_{gap}_ft"])

    

    df,extra_col=extrafeatures(df.copy())

    

    features = [

        #'Country_Region','Province_State',

        f"lag_{gap}_cc",

        f"lag_{gap}_ft",

        f"lag_{gap}_rc",

        "perc_1_ac",

        "perc_1_cc",

        "diff_1_cc",

        "diff_2_cc",

        "diff_3_cc",

        "diff_1_ft",

        "diff_2_ft",

        "diff_3_ft",

        "diff_123_cc",

        "diff_123_ft",

        "diff_change_1_cc",

        "diff_change_2_cc",

        "diff_change_1_ft",

        "diff_change_2_ft",

        "diff_change_12_cc",

        "diff_change_12_ft",

        "change_1_cc",

        "change_2_cc",

        "change_3_cc",

        "change_1_ft",

        "change_2_ft",

        "change_3_ft",

        "change_123_cc",

        "change_123_ft",

        "days_since_1_case",

        "days_since_10_case",

        "days_since_50_case",

        "days_since_100_case",

        "days_since_500_case",

        "days_since_1000_case",

        "days_since_5000_case",

        "days_since_10000_case",

        "country_flag",

        #"lat",

        #"lon",

        "continent",

        #"population",

        "area",

        "density",

        "target_cc",

        "target_ft"

    ]+extra_col

    df=df.sort_values(['Country_Region','Province_State','Date'])

    return df[features]

## function for building and predicting using LGBM model

def build_predict_lgbm(df_train, df_test, gap):

    LGB_PARAMS_C = {"objective": "regression",

              "num_leaves": 30,

              "learning_rate": 0.1,

              "bagging_fraction": 0.91,

              "feature_fraction": 0.71,

              #"min_data_in_leaf" : 50,

              #"max_bin":200,

              #"reg_alpha": 0.01,

              #"reg_lambda": 1,

              "metric": "rmse",

              "seed": SEED

             }

    

    LGB_PARAMS_F = {"objective": "regression",

              "num_leaves": 28,

              "learning_rate": 0.1,

              "bagging_fraction": 0.91,

              "feature_fraction": 0.4,

              #"min_data_in_leaf" : 50,

              #"max_bin":400,

              #"reg_alpha": 0.01,

              #"reg_lambda": 1,

              "metric": "rmse",

              "seed": SEED

             }

    

    df_train.dropna(subset = ["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace = True)

    

    target_cc = df_train.target_cc

    target_ft = df_train.target_ft

    

    test_lag_cc = df_test[f"lag_{gap}_cc"].values

    test_lag_ft = df_test[f"lag_{gap}_ft"].values

    

    df_train.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    df_test.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    

    categorical_features = ['continent','Regional_indicator']#,'Country_Region','Province_State']

    

    dtrain_cc = lgb.Dataset(df_train, label = target_cc, categorical_feature = categorical_features)

    dtrain_ft = lgb.Dataset(df_train, label = target_ft, categorical_feature = categorical_features)



    model_cc = lgb.train(LGB_PARAMS_C, train_set = dtrain_cc, num_boost_round = 1000)

    model_ft = lgb.train(LGB_PARAMS_F, train_set = dtrain_ft, num_boost_round = 1500)

    

    # inverse transform from log of change from last known value

    y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round = 1000) + np.log1p(test_lag_cc))

    y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round = 1500) + np.log1p(test_lag_ft))

    

    return y_pred_cc, y_pred_ft, model_cc, model_ft

## function for building and predicting using logistic

def build_predict_reg(df_train, df_test, gap):

    

    df_train.dropna(subset = ["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace = True)

    

    target_cc = df_train.target_cc

    target_ft = df_train.target_ft

    

    test_lag_cc = df_test[f"lag_{gap}_cc"].values

    test_lag_ft = df_test[f"lag_{gap}_ft"].values

    

    categorical_features = ['continent','Regional_indicator']#,'Country_Region','Province_State']

    

    df_train.drop(["target_cc", "target_ft"]+categorical_features, axis = 1, inplace = True)

    df_test.drop(["target_cc", "target_ft"]+categorical_features, axis = 1, inplace = True)

    

    df_train=df_train.replace(np.inf,np.nan)

    df_test=df_test.replace(np.inf,np.nan)

    for c in df_train.columns:

        df_train[c]=df_train[c].fillna(df_train[c].mean())

        df_test[c]=df_test[c].fillna(df_train[c].mean())

    model_cc=Ridge(alpha=300, fit_intercept=True, normalize=True, max_iter=1000, tol=0.0001, random_state=SEED)

    model_ft=Ridge(alpha=10, fit_intercept=True, normalize=True, max_iter=1000, tol=0.0001, random_state=SEED)

    

    model_cc.fit(df_train, target_cc)

    model_ft.fit(df_train, target_ft)

    

    # inverse transform from log of change from last known value

    y_pred_cc = np.expm1(model_cc.predict(df_test) + np.log1p(test_lag_cc))

    y_pred_ft = np.expm1(model_ft.predict(df_test) + np.log1p(test_lag_ft))

    

    return y_pred_cc, y_pred_ft, model_cc, model_ft

## building lag x-days models

df_train = df_panel[~df_panel.Id.isna()]

df_test_full = df_panel[~df_panel.ForecastId.isna()]



df_preds_val = []

df_preds_test = []



for pdate in df_test_full.Date.unique():

    

    print("Processing date:", pdate)

    

    # ignore date already present in train data

    if pdate in df_train.Date.values:

        df_pred_test = df_test_full.loc[df_test_full.Date == pdate, ["ForecastId", "ConfirmedCases", "Fatalities"]].rename(columns = {"ConfirmedCases": "ConfirmedCases_test", "Fatalities": "Fatalities_test"})

        

        # multiplying predictions by 41 to not look cool on public LB

        df_pred_test.ConfirmedCases_test = df_pred_test.ConfirmedCases_test * 41

        df_pred_test.Fatalities_test = df_pred_test.Fatalities_test * 41

    else:

        df_test = df_test_full[df_test_full.Date == pdate]

        

        gap = (pd.Timestamp(pdate) - max_date_train).days

        

        if gap <= VAL_DAYS:

            val_date = max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")



            df_build = df_train[df_train.Date < val_date]

            df_val = df_train[df_train.Date == val_date]

            

            X_build = prepare_features(df_build, gap)

            X_val = prepare_features(df_val, gap)

            

            #print('len of df_val{}, len of X_val{}'.format(len(df_val),len(X_val)) )

            y_val_cc_lgb, y_val_ft_lgb, _, _ = build_predict_lgbm(X_build.copy(), X_val.copy(), gap)

            y_val_cc_logy, y_val_ft_logy, _, _ = build_predict_reg(X_build.copy(), X_val.copy(), gap)         

            #y_val_cc_mad, y_val_ft_mad = predict_mad(df_val, gap, val = True)

            #print('{}_{}_{}'.format(len(df_val.Id.values),len(y_val_cc_lgb),len(y_val_ft_lgb)))

            df_pred_val = pd.DataFrame({"Id": df_val.Id.values,

                                        "ConfirmedCases_val_lgb": y_val_cc_lgb,

                                        "Fatalities_val_lgb": y_val_ft_lgb,

                                        "ConfirmedCases_val_logy": y_val_cc_logy,

                                        "Fatalities_val_logy": y_val_ft_logy,

                                       # "ConfirmedCases_val_mad": y_val_cc_mad,

                                      #  "Fatalities_val_mad": y_val_ft_mad,

                                       })



            df_preds_val.append(df_pred_val)



        X_train = prepare_features(df_train, gap)

        X_test = prepare_features(df_test, gap)



        y_test_cc_lgb, y_test_ft_lgb, model_cc, model_ft = build_predict_lgbm(X_train.copy(), X_test.copy(), gap)

        y_test_cc_logy, y_test_ft_logy, _, _ = build_predict_reg(X_train.copy(), X_test.copy(), gap)

       # y_test_cc_mad, y_test_ft_mad = predict_mad(df_test, gap)

        

        if gap == 1:

            model_1_cc = model_cc

            model_1_ft = model_ft

            features_1 = X_train.columns.values

            features_1=features_1[~np.isin(features_1,["target_cc", "target_ft"])]

        elif gap == 14:

            model_14_cc = model_cc

            model_14_ft = model_ft

            features_14 = X_train.columns.values

            features_14=features_14[~np.isin(features_14,["target_cc", "target_ft"])]

        elif gap == 28:

            model_28_cc = model_cc

            model_28_ft = model_ft

            features_28 = X_train.columns.values

            features_28=features_28[~np.isin(features_28,["target_cc", "target_ft"])]

        

        df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,

                                     "ConfirmedCases_test_lgb": y_test_cc_lgb,

                                     "Fatalities_test_lgb": y_test_ft_lgb,

                                     "ConfirmedCases_test_logy": y_test_cc_logy,

                                     "Fatalities_test_logy": y_test_ft_logy,

                                   #  "ConfirmedCases_test_mad": y_test_cc_mad,

                                  #   "Fatalities_test_mad": y_test_ft_mad,

                                    })

    

    df_preds_test.append(df_pred_test)

print(len(X_val.columns))
## validation score

df_panel = df_panel.merge(pd.concat(df_preds_val, sort = False), on = "Id", how = "left")

df_panel = df_panel.merge(pd.concat(df_preds_test, sort = False), on = "ForecastId", how = "left")



rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)))

rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities_val_lgb)))



rmsle_cc_logy = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_logy.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_logy.isna()].ConfirmedCases_val_logy)))

rmsle_ft_logy = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_logy.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_logy.isna()].Fatalities_val_logy)))



#rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)))

#rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities_val_mad)))



print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))

print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))

print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))

print("\n")

print("Logistic CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_logy, 2))

print("Logistic FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_logy, 2))

print("Logistic Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_logy + rmsle_ft_logy) / 2, 2))

#print("\n")

#print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))

#print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))

#print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))

## feature importance

from bokeh.io import output_notebook, show

from bokeh.layouts import column

from bokeh.palettes import Spectral3

from bokeh.plotting import figure



output_notebook()



df_fimp_1_cc = pd.DataFrame({"feature": features_1, "importance": model_1_cc.feature_importance(), "model": "m01"})

df_fimp_14_cc = pd.DataFrame({"feature": features_14, "importance": model_14_cc.feature_importance(), "model": "m14"})

df_fimp_28_cc = pd.DataFrame({"feature": features_28, "importance": model_28_cc.feature_importance(), "model": "m28"})



df_fimp_1_cc.sort_values("importance", ascending = False, inplace = True)

df_fimp_14_cc.sort_values("importance", ascending = False, inplace = True)

df_fimp_28_cc.sort_values("importance", ascending = False, inplace = True)



v1 = figure(plot_width = 800, plot_height = 400, x_range = df_fimp_1_cc.feature[:25], title = "Feature Importance of LGB Model 1")

v1.vbar(x = df_fimp_1_cc.feature[:25], top = df_fimp_1_cc.importance[:25], width = 1)

v1.xaxis.major_label_orientation = 1.3



v14 = figure(plot_width = 800, plot_height = 400, x_range = df_fimp_14_cc.feature[:25], title = "Feature Importance of LGB Model 14")

v14.vbar(x = df_fimp_14_cc.feature[:25], top = df_fimp_14_cc.importance[:25], width = 1)

v14.xaxis.major_label_orientation = 1.3



v28 = figure(plot_width = 800, plot_height = 400, x_range = df_fimp_28_cc.feature[:25], title = "Feature Importance of LGB Model 28")

v28.vbar(x = df_fimp_28_cc.feature[:25], top = df_fimp_28_cc.importance[:25], width = 1)

v28.xaxis.major_label_orientation = 1.3



v = column(v1, v14, v28)



show(v)

df_test = df_panel.loc[~df_panel.ForecastId.isna(), ["ForecastId", "Country_Region", "Province_State", "Date",

                                                     "ConfirmedCases_test", "ConfirmedCases_test_lgb",

                                                     "ConfirmedCases_test_logy","Fatalities_test_logy",

                                                     "Fatalities_test", "Fatalities_test_lgb"]].reset_index()

#df_test["ConfirmedCases"] = df_test.ConfirmedCases_test_lgb

#df_test["Fatalities"] = df_test.Fatalities_test_lgb



df_test["ConfirmedCases"] = 0.8 * df_test.ConfirmedCases_test_lgb + 0.2 * df_test.ConfirmedCases_test_logy

df_test["Fatalities"] = 0.8 * df_test.Fatalities_test_lgb + 0.2 * df_test.Fatalities_test_logy



df_test.loc[df_test.Date.isin(df_train.Date.values), "ConfirmedCases"] = df_test[df_test.Date.isin(df_train.Date.values)].ConfirmedCases_test.values

df_test.loc[df_test.Date.isin(df_train.Date.values), "Fatalities"] = df_test[df_test.Date.isin(df_train.Date.values)].Fatalities_test.values



df_submission = df_test[["ForecastId", "ConfirmedCases", "Fatalities"]]

df_submission.ForecastId = df_submission.ForecastId.astype(int)

df_submission.to_csv('submission.csv', index = False)