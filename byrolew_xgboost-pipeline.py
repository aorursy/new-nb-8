#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse as sp
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer




df_train = pd.read_csv('../input/train.csv')
df_macro = pd.read_csv('../input/macro.csv')
df_test = pd.read_csv('../input/test.csv')




# Join with macro variables
df_all = pd.concat([df_train, df_test])
df = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')




# Change what we can to floats
for col in df.columns:
    if df[col].dtype == int:
        df[col] = df[col].astype(float).copy()
    elif df[col].dtype != float:
        df.loc[df[col].str.contains('^no$', na=False), col] = 0.0
        df.loc[df[col].str.contains('^yes$', na=False), col] = 1.0
        try:
            df[col] = df[col].astype(float).copy()
        except ValueError:
            pass




# Change again to train and final test
df_all = df[np.isfinite(df['price_doc'])]
df_final_test = df[~ np.isfinite(df['price_doc'])]

x_final = df_final_test.drop(['price_doc', 'id'], axis=1)
y_final = df_final_test['price_doc']
id_test = df_final_test['id']

y_train = df_all['price_doc']
x_train = df_all.drop(['price_doc', 'id'], axis=1)




# Selector coumns by name or type
class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=None, columns=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or                (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return np.array(x[selected_cols[0]])
        else:
            return np.array(x[selected_cols])




# Converter fron string to int (for one hot encoder)
class StringConverter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.map = {} # column : string : int
    
    def fit(self, X, *args):
        for col in range(X.shape[1]):
            self.map[col] = {}
            idx = 1
            for row in range(X.shape[0]):                
                s = X[row, col]
                if s not in self.map[col]:
                    self.map[col][s] = idx
                    idx += 1
        return self

    def transform(self, X):
        X_int = np.zeros(shape=X.shape)
        for col in range(X.shape[1]):
            X_int[:, col] = np.array([self.map[col].get(s, 0) for s in X[:, col]])

        return X_int




# Adds a column in sparse matrix (because of bug in xgboost)
class AddDummy(BaseEstimator, TransformerMixin):
    def fit(self, X, *args):
        return self
    
    def transform(self, X):
        return sp.hstack([X, sp.csr_matrix(np.ones((X.shape[0], 1)))])




# Adds features based on date
class DatesFeaturer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.month_year_map = {}
        self.week_year_map = {}
    
    def fit(self, df, *args):
        month_year = pd.to_datetime(df.timestamp).dt.month + pd.to_datetime(df.timestamp).dt.year * 100
        self.month_year_map = month_year.value_counts().to_dict()
        week_year = pd.to_datetime(df.timestamp).dt.weekofyear + pd.to_datetime(df.timestamp).dt.year * 100
        self.week_year_map = week_year.value_counts().to_dict()
        return self
    
    def transform(self, df):
        month_year = pd.to_datetime(df.timestamp).dt.month + pd.to_datetime(df.timestamp).dt.year * 100
        week_year = pd.to_datetime(df.timestamp).dt.weekofyear + pd.to_datetime(df.timestamp).dt.year * 100
        
        new_df = pd.DataFrame({
            'month_year_count': month_year.map(self.month_year_map),
            'week_year_count': week_year.map(self.week_year_map),
            'month': pd.to_datetime(df.timestamp).dt.month,
            'dow': pd.to_datetime(df.timestamp).dt.dayofweek
        })
        
        return np.array(new_df)




# Adds relative features
class RelativeFeaturer(BaseEstimator, TransformerMixin):
    
    def fit(self, df, *args):
        return self
    
    def transform(self, df):   
        new_df = pd.DataFrame({
            'rel_floor': df['floor'] / np.maximum(1.0, df['max_floor'].astype(float)),
            'rel_kitch_sq': df['kitch_sq'] / np.minimum(1.0, df['full_sq'].astype(float)),
        })
        return np.array(new_df)




# Converts estimator to transform in order to ensemble many estimators
class EstimatorToTransform(BaseEstimator, TransformerMixin):    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, *args):
        self.estimator.fit(X, *args)
        return self

    def transform(self, X):
        pred = self.estimator.predict(X)
        return pred.reshape(-1, 1)




def rmsle_score(pred, true):
    return (np.sum((np.log(1 + pred) - np.log(1 + true))**2) / len(pred))**0.5




# Defining pipelines

float_pipeline = make_pipeline(
    PandasSelector(dtype=float),
)

eco_pipeline = make_pipeline(
    PandasSelector(columns=['ecology'], return_vector=False),
    StringConverter(),
    OneHotEncoder(),
)

prod_pipeline = make_pipeline(
    PandasSelector(columns=['product_type'], return_vector=False),
    StringConverter(),
    OneHotEncoder(),
)

sub_pipeline = make_pipeline(
    PandasSelector(columns=['sub_area'], return_vector=False),
    StringConverter(),
    OneHotEncoder(),
)

dates_pipeline = make_pipeline(
    DatesFeaturer(),
)

rel_pipeline = make_pipeline(
    RelativeFeaturer(),
)

pipeline_ensemble = make_pipeline(
    make_union(
        EstimatorToTransform(
            xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='reg:linear',
            ),
        ),
        EstimatorToTransform(
            xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=1,
                colsample_bytree=1,
                objective='reg:linear',
            ),
        ),
    ),
)

final_pipeline = make_pipeline(
    make_union(
        rel_pipeline,
        dates_pipeline,
        float_pipeline,
        eco_pipeline,
        prod_pipeline,
        sub_pipeline,
    ),
    AddDummy(),
    Imputer(),
    pipeline_ensemble,
    LinearRegression(),

)




# Calculate cross-validate score

cv_score = cross_val_score(
        final_pipeline,
        x_train,
        y_train,
        scoring=make_scorer(rmsle_score),
        cv=3,
)

np.mean(cv_score)




get_ipython().run_cell_magic('time', '', '# Fitting model\nmodel = final_pipeline.fit(x_train, y_train)')




pred_train = model.predict(x_train)
print(rmsle_score(pred_train, y_train))




# Predicting
final_pred = model.predict(x_final)




# Creating final submission file
df_sub = pd.DataFrame({'id': id_test.astype(int), 'price_doc': final_pred.astype(int)})
df_sub.to_csv('sub.csv', index=False)

