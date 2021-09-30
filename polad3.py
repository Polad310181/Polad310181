import pandas
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
data = pd.read_table (r'C:\Users\Casper\Desktop\housing.txt', delim_whitespace=True, header=None)
print(data.head(10))
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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense 
from tensorflow.keras import layers
#https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
model = Sequential()
model.add(Dense(13, kernel_initializer='uniform', input_dim=13, activation='relu'))
model.add(Dense(13, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, y_train, validation_split=0.33, epochs=100, batch_size=10)
model.fit(X_train, y_train, epochs=100, batch_size=10)
y_pred_Ysa=model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
r2_score(y_test, y_pred_Ysa)
mean_squared_error(y_test, y_pred_Ysa)
mean_absolute_error(y_test, y_pred_Ysa)
median_absolute_error(y_test, y_pred_Ysa)
for i in model.layers:
    ilk_gizli_katman=model.layers[0].get_weights()
    ikinci_gizli_katman=model.layers[1].get_weights()
    cikti_katman=model.layers[2].get_weights()