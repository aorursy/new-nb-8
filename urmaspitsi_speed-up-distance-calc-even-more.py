import numpy as np # linear algebra

import scipy as sp

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))



from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')



structures = pd.read_csv('../input/structures.csv')



def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)

# Engineer a single feature: distance vector between atoms

#  (there's ways to speed this up!)



def dist(row):

    return ( (row['x_1'] - row['x_0'])**2 +

             (row['y_1'] - row['y_0'])**2 +

             (row['z_1'] - row['z_0'])**2 ) ** 0.5



#train['dist'] = train.apply(lambda x: dist(x), axis=1)

#test['dist'] = test.apply(lambda x: dist(x), axis=1)

# 

# This block is SPPED UP



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



tr_a_min_b = train_p_0 - train_p_1

te_a_min_b = test_p_0 - test_p_1

train['dist_np_linalg'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist_np_linalg'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

# python vectorized: even faster!

train['dist_numpy'] = np.sqrt(np.sum((train_p_1 - train_p_0)**2, axis=1)) 

test['dist_numpy'] = np.sqrt(np.sum((test_p_1 - test_p_0)**2, axis=1))

train['dist_einsum'] = np.sqrt(np.einsum('ij,ij->i', tr_a_min_b, tr_a_min_b))

test['dist_einsum'] = np.sqrt(np.einsum('ij,ij->i', te_a_min_b, te_a_min_b))
train.head()