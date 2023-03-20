import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

sns.set_style('whitegrid')








df_train = pd.read_csv("../input/train.csv")

df_store = pd.read_csv("../input/store.csv")

df_test = pd.read_csv("../input/test.csv")





#Разделим дату на год и месяц

df_train['Year'] = df_train['Date'].apply(lambda x: int(x[:4]))

df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train.head()
#Ого, количество клиентов очень сильно коррелирует с продажами, почти 90 процентов

df_train['Customers']

df_train['Sales']

CaS = pd.DataFrame()

CaS['Customers'] = df_train['Customers']

CaS['Sales'] = df_train['Sales']

correlationMatrix = CaS.corr().abs()

plt.subplots(figsize=(13, 9))

sns.heatmap(correlationMatrix,annot=True)
#оказывается, каникулы и промо-акции влияют на продажи, на школьных каникулах продажи чуть больше, а в обычные

#их вообще нет

df_train["HolidayBin"] = df_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

sns.factorplot(x ="Year", y ="Sales", hue ="Promo", data = df_train,

                   size = 4, kind ="box", palette ="muted")

sns.factorplot(x ="Year", y ="Sales", hue ="SchoolHoliday", data = df_train,

                   size = 4, kind ="box", palette ="muted")

sns.factorplot(x ="Year", y ="Sales", hue ="HolidayBin", data = df_train,

                   size = 4, kind ="box", palette ="muted")
#разные типы каникул

df_train['StateHoliday'] = df_train['StateHoliday'].replace(0, '0')

df_train["HolidayBin"] = df_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

df_train.StateHoliday.unique()
#разберемся в структуре этих каникул

sns.factorplot(x ="Year", y ="Sales", hue ="StateHoliday", data = df_train, 

               size = 6, kind ="bar", palette ="muted")
average_customers = df_train.groupby('Month')["Customers"].mean()

average_sales = df_train.groupby('Month')['Sales'].mean()

total_sales_customers =  df_train.groupby('Store')['Sales', 'Customers'].sum()

total_sales_customers.head()
df_total_sales_customers = pd.DataFrame({'Sales':  total_sales_customers['Sales'],

                                         'Customers': total_sales_customers['Customers']}, 

                                         index = total_sales_customers.index)



df_total_sales_customers = df_total_sales_customers.reset_index()

df_total_sales_customers.head()
avg_sales_customers =  df_train.groupby('Store')['Sales', 'Customers'].mean()

avg_sales_customers.head()
df_avg_sales_customers = pd.DataFrame({'Sales':  avg_sales_customers['Sales'],

                                         'Customers': avg_sales_customers['Customers']}, 

                                         index = avg_sales_customers.index)



df_avg_sales_customers = df_avg_sales_customers.reset_index()



df_stores_avg = df_avg_sales_customers.join(df_store.set_index('Store'), on='Store')

df_stores_avg.head()

df_stores_new = df_total_sales_customers.join(df_store.set_index('Store'), on='Store')

df_stores_new.head()
#Больше всего посетителей и продаж в аптеках класса b

#Как это ни странно, но несмотря на то, что у магазинов класса b больше всего посетителей и продаж, 

#конкуренты находятся ближе всего к этому классу



average_storetype = df_stores_new.groupby('StoreType')['Sales', 'Customers', 'CompetitionDistance'].mean()



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,4))

sns.barplot(average_storetype.index, average_storetype['Sales'], ax=axis1)

sns.barplot(average_storetype.index, average_storetype['Customers'], ax=axis2)

sns.barplot(average_storetype.index, average_storetype['CompetitionDistance'], ax=axis3)



average_storetype.index
# В целом оказывается, что чем ближе конкурент, тем продажи ниже

Comp_Sales_Cust = pd.DataFrame()

Comp_Sales_Cust['Customers'] = average_storetype['Sales']

Comp_Sales_Cust['Sales'] = average_storetype['Customers']

Comp_Sales_Cust['Comp'] = average_storetype['CompetitionDistance']

correlationMatrix = Comp_Sales_Cust.corr()

plt.subplots(figsize=(13, 9))

sns.heatmap(correlationMatrix,annot=True)
#Посмотрим на ассортимент

average_assortment = df_stores_new.groupby('Assortment')['Sales', 'Customers'].mean()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(average_assortment.index, average_assortment['Sales'], ax=axis1)

sns.barplot(average_assortment.index, average_assortment['Customers'], ax=axis2)





df_train = pd.read_csv("../input/train.csv")

df_store = pd.read_csv("../input/store.csv")

df_test = pd.read_csv("../input/test.csv")



closed_store_ids = df_test["Id"][df_test["Open"] == 0].values

df_train['Year'] = df_train['Date'].apply(lambda x: int(x[:4]))

df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train["HolidayBin"] = df_train.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})

del df_train['Date']

del df_train['StateHoliday']

df_test['Year'] = df_test['Date'].apply(lambda x: int(x[:4]))

df_test['Month'] = df_test['Date'].apply(lambda x: int(x[5:7]))

df_test["HolidayBin"] = df_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})

del df_test['Date']

del df_test['StateHoliday']

df_test.head()
df_test = df_test[df_test["Open"] != 0]

df_test[df_test['Store'] == 1].head()

#CompetitionDistance

#df_test['CompetitionDistance'] =

a = list()

for i in df_test['Store']:

      a.append(float(df_store['CompetitionDistance'][df_store['Store'] == i]))

df_test['CompetitionDistance'] = a

a = list()

for i in df_train['Store']:

      a.append(float(df_store['CompetitionDistance'][df_store['Store'] == i]))

df_train['CompetitionDistance'] = a

df_train.head()

df_train['CompetitionDistance'] = df_train['CompetitionDistance'].fillna(df_train['CompetitionDistance'].mean())
#На всякий случай прологарифмируем данные, часто это помогает

df_train['CompetitionDistance'] = np.log(df_train['CompetitionDistance'])

df_test['CompetitionDistance'] = np.log(df_test['CompetitionDistance'])
#для каждой аптеки считаем свой отдельный алгоритм

#были попробованы разные виды регрессий, но случайны лес покажет наибольшую точность

#так как такие датафреймы получаются не очень большими, то можно использовать малое количество эстиматоров

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor





train_stores = dict(list(df_train.groupby('Store')))

test_stores = dict(list(df_test.groupby('Store')))

submission = pd.Series()



for i in test_stores:

    

   

    store = train_stores[i]

    X_train = store.drop(["Sales", "Store", "Customers"],axis=1)

    Y_train = store["Sales"]

    X_test  = test_stores[i].copy()



    

    store_ids = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    X_train = X_train.fillna(X_train.mean())

    X_test = X_test.fillna(X_train.mean())

    

    #RFR

    rfr = RandomForestRegressor(n_estimators = 5, criterion = 'mse')

    rfr.fit(X_train, Y_train)

    Y_pred = rfr.predict(X_test)

 

    



   

    submission = submission.append(pd.Series(Y_pred, index=store_ids))



submission = submission.append(pd.Series(0, index=closed_store_ids))

submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})

submission.to_csv('rossmann_submission.csv', index=False)



print (len(submission))