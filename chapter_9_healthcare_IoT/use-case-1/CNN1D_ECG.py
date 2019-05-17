"""
Chapter 9: Healthcare IoT

Code for CNN1D model training and testing on ECG 
"""

# Import the modules needed

from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import TensorBoard
import seaborn as sn
import matplotlib.pyplot as plt

np.random.seed(7)

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
samples_size= len(mats)
print('The number of total ECG samples is ', samples_size)

input_size = 10000
X = np.zeros((samples_size, input_size))


for i in range(samples_size):
    dummy = sio.loadmat(dataset_path + mats[i])['val'][0, :]
    if (input_size - len(dummy)) <= 0:
        X[i, :] = dummy[0:input_size]
    else:
        b = dummy[0:(input_size - len(dummy))]
        goal = np.hstack((dummy, b))
        while len(goal) != input_size:
            b = dummy[0:(input_size - len(goal))]
            goal = np.hstack((goal, b))
        X[i, :] = goal

target_train = np.zeros((samples_size, 1))
Train_data = pd.read_csv(dataset_path + 'REFERENCE.csv', sep=',', header=None, names=None)
for i in range(samples_size):
    if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
        target_train[i] = 0
    elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
        target_train[i] = 1
    elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
        target_train[i] = 2
    else:
        target_train[i] = 3

Label_set = np.zeros((samples_size, number_of_classes))

for i in range(samples_size):
    dummy = np.zeros((number_of_classes))
    dummy[int(target_train[i])] = 1
    Label_set[i, :] = dummy

X = (X - X.mean())/(X.std()) # Normalization 
X = np.expand_dims(X, axis=2) #For Keras's data input size


values = [i for i in range(samples_size)]
permutations = np.random.permutation(values)
X = X[permutations, :]
Label_set = Label_set[permutations, :]

#Training data preparation
train = 0.9 # Fraction of the dataset for training 
X_train = X[:int(train * samples_size), :]
Y_train = Label_set[:int(train * samples_size), :]
X_val = X[int(train * samples_size):, :]
Y_val = Label_set[int(train * samples_size):, :]


# CNN1 DL model cration 
model = Sequential()
model.add(Conv1D(128, 55, activation='relu', input_shape=(input_size, 1)))
model.add(MaxPooling1D(10))
model.add(Dropout(0.5))
model.add(Conv1D(128, 25, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))
model.add(Conv1D(128, 10, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalAveragePooling1D())

model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))

# Logging through Tensorbard
tensorboard = TensorBoard(log_dir='./logs_cnn',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
# Model compilation 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Model saving for transfer learning 
checkpointer = ModelCheckpoint(filepath='trained-models-cnn/Best_model.h5', monitor='val_acc', verbose=1, save_best_only=True)
# Model fitting
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=500, verbose=2, shuffle=True, callbacks=[checkpointer, tensorboard])
pd.DataFrame(hist.history).to_csv(path_or_buf='trained-models-cnn/History.csv')

# Model validation
predictions = model.predict(X_val)
score = accuracy_score(change_format(Y_val), change_format(predictions))
print('Last epoch\'s validation score is ', score)

# Storing Results to file 
df = pd.DataFrame(change_format(predictions))
df.to_csv(path_or_buf='trained-models-cnn/Preds_' + str(format(score, '.4f')) + '.csv', index=None, header=None)
pd.DataFrame(confusion_matrix(change_format(Y_val), change_format(predictions))).to_csv(path_or_buf='trained-models-cnn/Result_Conf' + str(format(score, '.4f')) + '.csv', index=None, header=None)

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
plt.savefig('figures/confusion-matrix-cnn-ecg.png', dpi=300)
plt.show()
	
	

