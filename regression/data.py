# !/usr/bin/python

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

df = quandl.get('WIKI/GOOGL', api_key='PYVBdRxzpwCCch2p-Z3A') #Get Stock Statistic Data From Quandl
# quandl.ApiConfig.api_base = 'PYVBdRxzpwCCch2p-Z3A'
# print(df.head())
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) #0.01 is 10 Percent
# print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)
# print(df.tail())

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)

x = x[:-forecast_out]
x_lately = x[-forecast_out:]

df.dropna(inplace=True)

y = np.array(df['label'])
y = np.array(df['label'])

# X is Feature / Price
# Y is Labeling

#Start Training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# clf = LinearRegression()
# clf = svm.SVR(kernel='poly')
clf = LinearRegression(n_jobs=5)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
# print(accuracy)
#End Training, post Number of Accuracy

# Predict
forecast_set = clf.predict(x_lately)

# print(forecast_set, accuracy, forecast_out)
# End Predict

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()