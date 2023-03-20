# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load train and test dataset 

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#printing train dataset information 

train.info()
test.info()
pd.set_option('display.max_columns',1000)

pd.set_option('display.max_rows',1000)
train.describe()