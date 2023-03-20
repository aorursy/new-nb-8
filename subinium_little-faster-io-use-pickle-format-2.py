import os

import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


train_pickle = pd.read_pickle('/kaggle/input/little-faster-io-use-pickle-format/train.pkl')

train_pickle.head()