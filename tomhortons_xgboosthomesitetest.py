import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

seed = 260681

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
test['Date'].dt.dayofweek
test.columns