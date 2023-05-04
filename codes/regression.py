import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# read df
F = open(r'df.pkl', 'rb')
df = pickle.load(F)

# resample it into trading days
df = df.resample('B').sum()

# drop the useless data
start_date = pd.Timestamp('2011-01-04')
end_date = pd.Timestamp('2022-01-01')

to_keep = df.loc[(df.index >= start_date) & (df.index <= end_date)]
df = pd.DataFrame(to_keep)

# import daily return
teslareturn = pd.read_csv('dailyreturn.csv')

# reform data into dataframe
teslareturn = pd.DataFrame(teslareturn)
teslareturn['date'] = pd.to_datetime(teslareturn['date'])
teslareturn = teslareturn.set_index('date')

# drop the useless data
to_keep = teslareturn.loc[(teslareturn.index >= start_date) & (teslareturn.index <= end_date)]
teslareturn = pd.DataFrame(to_keep)

# pair the date index with df and teslareturn
df = df.reindex(teslareturn.index)
df = df.fillna(method='ffill')


# replace zero with array[0,0,0,0,0,0]
def replace_zero_with_array(x):
    if isinstance(x, int) and x == 0:
        return np.array([0, 0, 0, 0, 0, 0])
    else:
        return x


df = df.applymap(replace_zero_with_array)

# separate them to regress separately
df1 = pd.DataFrame(df['sentiments'])
df2 = pd.DataFrame(df['accumulated'])

# split the array
df1[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']] = pd.DataFrame(df1['sentiments'].tolist(), index=df1.index)
df1 = df1.drop('sentiments', axis=1)

df2[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']] = pd.DataFrame(df2['accumulated'].tolist(), index=df1.index)
df2 = df2.drop('accumulated', axis=1)

# combine to do the regression
df = pd.concat([df1, teslareturn], axis=1)

# check VIF and correlations
X = df.iloc[:, :-1]
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

correlations = df.corr()['tesla']
print(correlations)

# regression - 10-year-period
X = sm.add_constant(X)
model = sm.OLS(df['tesla'], X).fit()
print(model.summary())


# before the covid
df_pre_covid = df.loc[(df.index >= '2011-01-01') & (df.index <= '2019-12-31')]

# regression - before the covid
X = sm.add_constant(X)
model = sm.OLS(df_pre_covid['tesla'], X).fit()
print(model.summary())

# during the covid
df_during_covid = df.loc[(df.index >= '2020-01-01') & (df.index <= '2021-12-31')]

# regression - during the covid
X = sm.add_constant(X)
model = sm.OLS(df_during_covid['tesla'], X).fit()
print(model.summary())


