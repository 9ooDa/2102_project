import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv')
df = df.dropna()
le = LabelEncoder()
df['cbwd'] = le.fit_transform(df['cbwd'])
df = df.iloc[:,1:] # remove row number column

# convert to datetime
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']], format='%Y/%m/%d, %H')
df = df.drop(['year','month','day','hour'], axis = 1)
df = df.reset_index()
df = df[['datetime','pm2.5','DEWP','PRES','cbwd','Iws','Is','Ir']]

df['datetime'] = pd.to_numeric(pd.to_datetime(df['datetime']))
dataset = df.values
X = dataset[:,[0,2,3,4,5,6,7]]
Y = dataset[:,1]
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
# X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# fit final model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=100,
           max_features='sqrt', min_samples_leaf=4,
           min_samples_split=6, n_estimators=100)
model.fit(X_train, Y_train)
ynew = model.predict(X_test)

# for i in range(len(X_test)):
#     print("X= {}, True_Y= {} ,Predicted= {}".format(X_test[i], Y_test[i] ,ynew[i]))

print("Accuracy Score:", model.score(X_test,Y_test))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(estimator = model, X = X_train, y = Y_train, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)

from sklearn.metrics import mean_squared_error
from math import sqrt
print('RMSE:',sqrt(mean_squared_error(Y_test, ynew))) 

from numpy import mean
from numpy import absolute
from numpy import sqrt
print('RMSE(10-fold cross-validation):',sqrt(mean(absolute(scores))))