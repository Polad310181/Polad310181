import pandas
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
data = pd.read_table (r'C:\Users\Casper\Desktop\housing.txt', delim_whitespace=True, header=None)
#value değere dönüştürmeden önce X y ayırmak olmur 
#veriseti=data
#veriseti.info (bu kontrol value değere dönüştürmeden önce yapıla bilir ve data
# ve data frame dair bilgi verir kaç satır kaç sutun). ya x y ayırarken ya önce value deyere 
# deyere ayırmak lazım yada 
veriseti=data.values
X = veriseti[:,0:13]
y=veriseti[:,13]
#X= veriseti.iloc[:,:-1].values #(yadaburada olduğu gibi ayırarken yapırız)
#y=veriseti.iloc[:,-1].values
#y=y.astype("float64")#floata dönüştürme işlemi bu şekilde olur.
#X=X.astype("float64")#her zaman gerek olmur kendim öğlesine yazdım
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.30, random_state=0)
#X_train=X_train.astype("float64") #buda floata dönüştürme işlemi
from sklearn.preprocessing import MinMaxScaler 
scaler =MinMaxScaler() 
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
y_train=scaler.fit_transform(y_train.reshape(-1,1))
y_test=scaler.fit_transform(y_test.reshape(-1,1))
###################
#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
###################################
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
######################
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#performans değerlendirme
#polad
