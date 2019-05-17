"""
Chapter 9: Healthcare IoT

Code for LSTM model training and testing on ECG data
"""

# Import the modules needed

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
import seaborn as sn

#Number of classes: [normal rhythm, atrial fibrillation, other rhythm, noisy measurements]
number_of_classes = 4 

# Function change format: from boolean arrays to decimal arrays
def change_format(x): 
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)

#Data loading and pre-processing

dataset_path = 'dataset/ECG/'
# Read all files from the folder linked with measurements, means start with 'A' 
all_files = [f for f in listdir(dataset_path) if (isfile(join(dataset_path, f)) and f[0] == 'A')]
bats = [f for f in all_files if f[7] == 'm']
input_size_threshold = 9000 
mats = [f for f in bats if (np.shape(sio.loadmat(dataset_path + f)['val'])[1] >= input_size_threshold)]
check = np.shape(sio.loadmat(dataset_path + mats[0])['val'])[1]
X = np.zeros((len(mats), check))
for i in range(len(mats)):
    X[i, :] = sio.loadmat(dataset_path + mats[i])['val'][0, :input_size_threshold]

target_train = np.zeros((len(mats), 1))
Train_data = pd.read_csv(dataset_path + 'REFERENCE.csv', sep=',', header=None, names=None)
for i in range(len(mats)):
    if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
        target_train[i] = 0
    elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
        target_train[i] = 1
    elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
        target_train[i] = 2
    else:
        target_train[i] = 3

Label_set = np.zeros((len(mats), number_of_classes))
for i in range(np.shape(target_train)[0]):
    dummy = np.zeros((number_of_classes))
    dummy[int(target_train[i])] = 1
    Label_set[i, :] = dummy


#Training data preparation
train_len = 0.9 # Fraction of the dataset for training 
X_train = X[:int(train_len*len(mats)), :]
Y_train = Label_set[:int(train_len*len(mats)), :]
X_val = X[int(train_len*len(mats)):, :]
Y_val = Label_set[int(train_len*len(mats)):, :]

# reshape input to be [samples, time steps, features]
X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = numpy.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# LSTM DL model cration 
batch_size = 32
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1, check)))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(number_of_classes, activation='softmax'))

# Logging through Tensorbard
tensorboard = TensorBoard(log_dir='./logs_lstm',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# Model compilation 
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='trained-models-lstm/Best_model.h5', monitor='val_acc', verbose=1, save_best_only=True)
# Model fitting
model.fit(X_train, Y_train, epochs=500, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2, shuffle=False, callbacks=[checkpointer, tensorboard])
# Model validation
predictions = model.predict(X_val)
score = accuracy_score(change_format(Y_val), change_format(predictions))
print('Last epoch\'s validation score is ', score)

# Confusion Matrix generation and plotting
confusion_matrix=pd.DataFrame(confusion_matrix(change_format(Y_val), change_format(predictions)), index = [i for i in "0123"],
                  columns = [i for i in "0123"])
confusion_matrix_normalised= confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

print(confusion_matrix)
print(confusion_matrix_normalised)
plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix_normalised, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.savefig('figures/confusion-matrix-lstm-ecg.png', dpi=300)
plt.show()
