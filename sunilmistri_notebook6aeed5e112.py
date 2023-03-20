# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print ("file Sizes")

for filename in os.listdir('../input'):

    print (filename.ljust(50)  + str( round(os.path.getsize('../input/' + filename)/1000000, 2))  + ' MB')

test  = pd.read_csv('../input/clicks_test.csv')

train = pd.read_csv('../input/clicks_train.csv')

sizes_train = train.groupby('display_id')['ad_id'].count().value_counts()

print(sizes_train)