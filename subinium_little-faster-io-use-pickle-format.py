import pandas as pd

import pickle

PATH = '../input/data-science-bowl-2019'

train_df = pd.read_csv(PATH+'/train.csv')

train_df.to_pickle("./train.pkl")

train_pickle = pd.read_pickle('./train.pkl')