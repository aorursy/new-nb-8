import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')
target = train['target']

train_id = train['id']

test_id = test['id']



train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
df = pd.concat([train, test], axis=0, sort=False )
bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}

df['bin_3'] = df['bin_3'].map(bin_dict)

df['bin_4'] = df['bin_4'].map(bin_dict)

print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],

                    prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], 

                    drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
from pandas.api.types import CategoricalDtype 



ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)



df.ord_1 = df.ord_1.astype(ord_1)

df.ord_2 = df.ord_2.astype(ord_2)

df.ord_3 = df.ord_3.astype(ord_3)

df.ord_4 = df.ord_4.astype(ord_4)



df.ord_1 = df.ord_1.cat.codes

df.ord_2 = df.ord_2.cat.codes

df.ord_3 = df.ord_3.cat.codes

df.ord_4 = df.ord_4.cat.codes

def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



df = date_cyc_enc(df, 'day', 7)

df = date_cyc_enc(df, 'month', 12)

from sklearn.preprocessing import LabelEncoder



# Label Encoding

for f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']:

    lbl = LabelEncoder()

    lbl.fit(df[f])

    df[f'le_{f}'] = lbl.transform(df[f])


df.drop(['nom_5','nom_6','nom_7','nom_8','nom_9', 'ord_5'] , axis=1, inplace=True)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
df = reduce_mem_usage(df)
train = df[:train.shape[0]]

test = df[train.shape[0]:]



train.shape

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    train, target, test_size=0.2, random_state=2019

)



xgb_clf = XGBClassifier(learning_rate=0.05,

    n_estimators=2000,

    seed=42,

    eval_metric='auc',

)



xgb_clf.fit(

    X_train, 

    y_train, 

    eval_set=[(X_train, y_train), (X_val, y_val)],

    early_stopping_rounds=20,

    verbose=20

)



results = xgb_clf.evals_result()

epochs = len(results['validation_0']['auc'])

x_axis = range(0, epochs)



# plot log loss

plt.figure(figsize=(15, 7))

plt.plot(x_axis, results['validation_0']['auc'], label='Train')

plt.plot(x_axis, results['validation_1']['auc'], label='Val')

plt.legend()

plt.ylabel('AUC')

plt.xlabel('# of iterations')

plt.title('XGBoost AUC')

plt.show()

import shap



shap.initjs()



explainer = shap.TreeExplainer(xgb_clf)

shap_values = explainer.shap_values(train)
shap.summary_plot(shap_values, train, plot_type="bar")
shap.summary_plot(shap_values, train)
shap.dependence_plot("le_ord_5", shap_values, train)
shap.dependence_plot("ord_4", shap_values, train)
shap.dependence_plot("day", shap_values, train)
shap.force_plot(explainer.expected_value, shap_values[0,:], train.iloc[0,:])