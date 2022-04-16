import pandas as pd
import numpy as np

df = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv')

df = df.dropna()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['cbwd'] = le.fit_transform(df['cbwd'])

df = df.iloc[:,1:] # remove row number column

# convert to datetime
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']], format='%Y/%m/%d, %H')
df = df.drop(['year','month','day','hour'], axis = 1)
df = df.reset_index()
df = df[['datetime','pm2.5','DEWP','PRES','cbwd','Iws','Is','Ir']]


from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

print(df)

df['datetime'] = pd.to_numeric(pd.to_datetime(df['datetime']))

print(df)

dataset = df.values
# X = dataset[:,[0,1,2,3,5,6,7,8,9,10,11]]


X = dataset[:,[0,2,3,4,5,6,7]]
Y = dataset[:,1]

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)
dtree.fit(X_train, Y_train)

pred_train_tree= dtree.predict(X_train)
print(np.sqrt(mean_squared_error(Y_train,pred_train_tree)))
print(r2_score(Y_train, pred_train_tree))

# Code lines 4 to 6
pred_test_tree= dtree.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test,pred_test_tree))) 
print(r2_score(Y_test, pred_test_tree))