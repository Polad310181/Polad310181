veriseti = pd.read_csv (r'C:\Users\Casper\Desktop\Ann.csv', sep=",")
veriseti
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
print(veriseti.head(10))
import re
kayip_veriler=[]
sayisal_olmayan_veriler=[]
for oznitelik in veriseti:
    #tüm eşsiz öznitelik değerleri
    essis_deger=veriseti[oznitelik].unique()
    print("'{}' özniteliklerine ait essis(unique) veriler {}" .format(oznitelik, essis_deger.size))
    if (essis_deger.size>10):
        print("10 adet benzersiz değer listele")
    print(essis_deger[0:10])
    print("\n-------------------------------------------------------------\n")
    #Özniteliklere ait sayısal olmayan veriyi bul
    if (True in pd.isnull(essis_deger)):
        s="{} özelliğine ait kayıp veriler {}" .format(oznitelik, 
                                                       pd.isnull(veriseti[oznitelik]).sum())
        kayip_veriler.append(s)
     #Özniteliklere ait sayısal olmayan verileri bul
    for i in range(0, np.prod(essis_deger.shape)):
         if(re.match('nan', str(essis_deger[i]))):
            break
         if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(essis_deger[i]))):
             sayisal_olmayan_veriler.append(oznitelik)
             break
print("Kayıp veriye sahip öznitelikler: \n{}\n\n".format(kayip_veriler))
print("Sayısal olmayan veriye sahip öznitelikler:\n{}".format(sayisal_olmayan_veriler))
 #eğitim ve test verisetini ayarla
 X= veriseti.iloc[:,:-1].values
 y=veriseti.iloc[:,5].values
 y
 X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=0)
 #öznitelik ölçekleme sistemi
from sklearn.preprocessing import MinMaxScaler 
scaler =MinMaxScaler() 
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
y_train=scaler.fit_transform(y_train.reshape(-1,1))
y_test=scaler.fit_transform(y_test.reshape(-1,1))
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras import layers
#https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
model = Sequential()
model.add(Dense(6, kernel_initializer='uniform', input_dim=5, activation='relu'))
model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)
#model.fit(X_train, y_train, validation_split=0.33, epochs=100, batch_size=10)
y_pred_Ysa=model.predict(X_test)
for i in model.layers:
    ilk_gizli_katman=model.layers[0].get_weights()
    ikinci_gizli_katman=model.layers[1].get_weights()
    cikti_katman=model.layers[2].get_weights()
#performans değerlendirme
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
r2_score(y_test, y_pred_Ysa)
mean_squared_error(y_test, y_pred_Ysa)
mean_absolute_error(y_test, y_pred_Ysa)
median_absolute_error(y_test, y_pred_Ysa)    


############################
#siniflandirici=Sequential()
#ilk gizli katman
#siniflandirici.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim = 5))
#ikinci gizli katman
#siniflandirici.add(Dense(6, kernel_initializer='uniform', activation='relu'))
#çıktı katmanı
#siniflandirici.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#derleme işlemi
#siniflandirici.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accurasy'])
#parametre seçiminden sonra Eğitim setinin yapay sınır ağlarına göre uyduruması
#import tensorflow as tf
#y_train_one_hot = tf.one_hot(y_train, depth=10)
#siniflandirici.fit(X_train, y_train_one_hot, batch_size=100, epochs=10)
#siniflandirici.fit(X_train, y_train, epochs=100, batch_size=10)
#y_pred_Ysa=siniflandirici.predict(X_test)
#ağırlıklar
#for i in siniflandirici.layers:
#  ilk_gizli_katman=siniflandirici.layers[0].get_weights()
#  ikinci_gizli_katman=siniflandirici.layers[1].get_weights()
#  cikti_katman=siniflandirici.layers[2].get_weights()
    