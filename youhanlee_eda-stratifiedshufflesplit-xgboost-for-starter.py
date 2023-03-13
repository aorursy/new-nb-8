import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib




import seaborn as sns # visualization

import missingno as msno



import xgboost as xgb # Gradient Boosting

import warnings

sns.set(style='white', context='notebook', palette='deep')



# warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output
np.random.seed(1989)

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train shape : ", train.shape)

print("Test shape : ", test.shape )
train.head()
print(train.info())
print(test.info())
targets = train['target'].values
sns.set(style="darkgrid")

ax = sns.countplot(x = targets)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))

plt.title('Distribution of Target', fontsize=20)

plt.xlabel('Claim', fontsize=20)

plt.ylabel('Frequency [%]', fontsize=20)

ax.set_ylim(top=700000)
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('Oh no')

print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values)) == 0 else print('Oh no')

print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] else print('Oh no')
import missingno as msno



train_null = train

train_null = train_null.replace(-1, np.NaN)



msno.matrix(df=train_null.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
test_null = test

test_null = test_null.replace(-1, np.NaN)



msno.matrix(df=test_null.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
# Extract columns with null data

train_null = train_null.loc[:, train_null.isnull().any()]

test_null = test_null.loc[:, test_null.isnull().any()]



print(train_null.columns)

print(test_null.columns)
print('Columns \t Number of NaN')

for column in train_null.columns:

    print('{}:\t {}'.format(column,len(train_null[column][np.isnan(train_null[column])])))
# divides all features in to 'bin', 'cat' and 'etc' group.



feature_list = list(train.columns)

def groupFeatures(features):

    features_bin = []

    features_cat = []

    features_etc = []

    for feature in features:

        if 'bin' in feature:

            features_bin.append(feature)

        elif 'cat' in feature:

            features_cat.append(feature)

        elif 'id' in feature or 'target' in feature:

            continue

        else:

            features_etc.append(feature)

    return features_bin, features_cat, features_etc



feature_list_bin, feature_list_cat, feature_list_etc = groupFeatures(feature_list)

print("# of binary feature : ", len(feature_list_bin))

print("# of categorical feature : ", len(feature_list_cat))

print("# of other feature : ", len(feature_list_etc))
def TrainTestHistogram(train, test, feature):

    fig, axes = plt.subplots(len(feature), 2, figsize=(10, 40))

    fig.tight_layout()



    left  = 0  # the left side of the subplots of the figure

    right = 0.9    # the right side of the subplots of the figure

    bottom = 0.1   # the bottom of the subplots of the figure

    top = 0.9      # the top of the subplots of the figure

    wspace = 0.3   # the amount of width reserved for blank space between subplots

    hspace = 0.7   # the amount of height reserved for white space between subplot



    plt.subplots_adjust(left=left, bottom=bottom, right=right, 

                        top=top, wspace=wspace, hspace=hspace)

    count = 0

    for i, ax in enumerate(axes.ravel()):

        if i % 2 == 0:

            title = 'Train: ' + feature[count]

            ax.hist(train[feature[count]], bins=30, normed=False)

            ax.set_title(title)

        else:

            title = 'Test: ' + feature[count]

            ax.hist(test[feature[count]], bins=30, normed=False)

            ax.set_title(title)

            count = count + 1
# For bin features

TrainTestHistogram(train, test, feature_list_bin)
# For cat features

TrainTestHistogram(train, test, feature_list_cat)
# For etc features

TrainTestHistogram(train, test, feature_list_etc)
left  = 0  # the left side of the subplots of the figure

right = 0.9    # the right side of the subplots of the figure

bottom = 0.1   # the bottom of the subplots of the figure

top = 0.9      # the top of the subplots of the figure

wspace = 0.3   # the amount of width reserved for blank space between subplots

hspace = 0.7   # the amount of height reserved for white space between subplot



fig, axes = plt.subplots(13, 2, figsize=(10, 40))

plt.subplots_adjust(left=left, bottom=bottom, right=right, 

                    top=top, wspace=wspace, hspace=hspace)



for i, ax in enumerate(axes.ravel()):

    title = 'Train: ' + feature_list_etc[i]

    ax.hist(train[feature_list_etc[i]], bins=20, normed=True)

    ax.set_title(title)

    ax.text(0, 1.2, train[feature_list_etc[i]].head(), horizontalalignment='left',

            verticalalignment='top', style='italic',

       bbox={'facecolor':'red', 'alpha':0.2, 'pad':10}, transform=ax.transAxes)
# For ordinal group

etc_ordianal_features = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01',

                    'ps_reg_02', 'ps_car_11', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03',

                    'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',

                    'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',

                    'ps_calc_14']



etc_continuous_features = ['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']



train_null_columns = train_null.columns

test_null_columns = test_null.columns
# For train

for feature in train_null_columns:

    if 'cat' in feature or 'bin' in feature:

        # For categorical and binary features with postfix, substitue null values with the most frequent value to avoid float number.

        train_null[feature].fillna(train_null[feature].value_counts().idxmax(), inplace=True)

    elif feature in etc_continuous_features:

        train_null[feature].fillna(train_null[feature].median(), inplace=True)

    elif feature in etc_ordianal_features:

        # For categorical and binary features which was assumed, substitue null values with the most frequent value to avoid float number.

        train_null[feature].fillna(train_null[feature].value_counts().idxmax(), inplace=True)

    else:

        print(feature)
# For test

for feature in test_null_columns:

    if 'cat' in feature or 'bin' in feature:

        # For categorical and binary features with postfix, substitue null values with the most frequent value to avoid float number.

        test_null[feature].fillna(test_null[feature].value_counts().idxmax(), inplace=True)

    elif feature in etc_continuous_features:

        test_null[feature].fillna(test_null[feature].median(), inplace=True)

    elif feature in etc_ordianal_features:

        # For categorical and binary features which was assumed, substitue null values with the most frequent value to avoid float number.

        test_null[feature].fillna(test_null[feature].value_counts().idxmax(), inplace=True)

    else:

        print(feature)
for feature in train_null_columns:

    train[feature] = train_null[feature]

    

for feature in test_null_columns:

    test[feature] = test_null[feature]
msno.matrix(df=train.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
msno.matrix(df=test.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))
def oneHotEncode_dataframe(df, features):

    for feature in features:

        temp_onehot_encoded = pd.get_dummies(df[feature])

        column_names = ["{}_{}".format(feature, x) for x in temp_onehot_encoded.columns]

        temp_onehot_encoded.columns = column_names

        df = df.drop(feature, axis=1)

        df = pd.concat([df, temp_onehot_encoded], axis=1)

    return df
train = oneHotEncode_dataframe(train, feature_list_cat)

test = oneHotEncode_dataframe(test, feature_list_cat)
# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
n_split = 3

SSS = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=1989)
# Parameter optimization is needed!

params = {

    'min_child_weight': 10.0,

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

}
X = train.drop(['id', 'target'], axis=1).values

y = train.target.values

test_id = test.id.values

test = test.drop('id', axis=1)
sub = pd.DataFrame()

sub['id'] = test_id

sub['target'] = np.zeros_like(test_id)
SSS.get_n_splits(X, y)
print(SSS)
for train_index, test_index in SSS.split(X, y):

    print("TRAIN: ", train_index, "TEST: ", test_index)
for i, (train_index, test_index) in enumerate(SSS.split(X, y)):

    print('------# {} of {} shuffle split------'.format(i + 1, n_split))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    

    # Convert splited data into XGBoost format

    d_train = xgb.DMatrix(X_train, y_train)

    d_valid = xgb.DMatrix(X_valid, y_valid)

    d_test = xgb.DMatrix(test.values)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]



    # Train the model! 

    model = xgb.train(params, d_train, 2000, watchlist, 

                      early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=100)



    print('------# {} of {} prediction------'.format(i + 1, n_split))

    # Predict on our test data

    p_test = model.predict(d_test)

    sub['target'] = sub['target'] + p_test/n_split
# sub.to_csv('stratifiedShuffleSplit_xgboost.csv', index=False)