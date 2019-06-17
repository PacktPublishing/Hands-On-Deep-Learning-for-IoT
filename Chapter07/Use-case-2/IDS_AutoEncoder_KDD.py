#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:21:25 2019

@author: raz
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pylab as plt


## For OMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Functions for DataLoading and pre-processing

def DataLoading (mypath):
    print ("Loading the data")
    dataframe = pd.read_csv(mypath,header = None,engine = 'python',sep=",")
    return dataframe

def DataPreprocessing(mydataframe):
    
    # Dropping the duplicates
    recordcount = len(mydataframe)
    print ("Original number of records in the training dataset before removing duplicates is: " , recordcount)
    mydataframe.drop_duplicates(subset=None, inplace=True)  # Python command to drop duplicates
    newrecordcount = len(mydataframe)
    print ("Number of records in the training dataset after removing the duplicates is :", newrecordcount,"\n")

    #Dropping the labels to a different dataset which is used to train the recurrent neural network classifier
    df_X = mydataframe.drop(mydataframe.columns[41],axis=1,inplace = False)
    df_Y = mydataframe.drop(mydataframe.columns[0:41],axis=1, inplace = False)

    # Convert Categorial data to the numerical data for the efficient classification
    df_X[df_X.columns[1:4]] = df_X[df_X.columns[1:4]].stack().rank(method='dense').unstack()
    
    # Coding the normal as " 1 0" and attack as "0 1"
    df_Y[df_Y[41]!='normal.'] = 0
    df_Y[df_Y[41]=='normal.'] = 1
    #print (labels[41].value_counts())
    
    #converting input data into float which is requried in the future stage of building in the network
    df_X = df_X.loc[:,df_X.columns[0:41]].astype(float)

    # Normal is "1 0" and the abnormal is "0 1"
    df_Y.columns = ["y1"]
    df_Y.loc[:,('y2')] = df_Y['y1'] ==0
    df_Y.loc[:,('y2')] = df_Y['y2'].astype(int)
    
    return df_X,df_Y


print ("Laoding the IDS Data")
#data_path = "Final_App_Layer.txt"
#data_path = "Final_Transport_Layer.txt"
data_path = "Final_Network_Layer.txt"

dataframe = DataLoading(data_path)

print ("Data Preprocessing of loaded IDS Data")
data_X, data_Y = DataPreprocessing(dataframe)


##### Function for features selection for the Model Training

def FeatureSelection(myinputX, myinputY):

    labels = np.array(myinputY).astype(int)
    inputX = np.array(myinputX)
    
    #Random Forest Model
    model = RandomForestClassifier(random_state = 0)
    model.fit(inputX,labels)
    importances = model.feature_importances_
    
    
    #Plotting the Features agains their importance scores
    indices = np.argsort(importances)[::-1]
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    plt.figure(figsize = (10,5))
    plt.title("Feature importances (y-axis) vs Features IDs(x-axis)")
    plt.bar(range(inputX.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(inputX.shape[1]), indices)
    plt.xlim([-1, inputX.shape[1]])
    plt.show()
    
    # Selecting top featueres which have higher importance values = here we can find "12" features
    #as we can see in the next step
    newX = myinputX.iloc[:,model.feature_importances_.argsort()[::-1][:12]]
   # Converting the X dataframe into tensors
    myX = newX.as_matrix()
    myY = labels

    return myX,myY

## Visualise the data for feature selction
    

print ("Performing the Feature Selection on train data set")
reduced_X,reduced_Y = FeatureSelection(data_X,data_Y)

#Dividing the dataset into train and test datasets
# Out of 13051 samples = 80% samples as train data and 20% samples as test data

# Train features and Train Labels
train_X = reduced_X[:8000]
train_Y = reduced_Y[:8000]

#Test Features and Test Labels
test_X = reduced_X[8001:10000]
test_Y = reduced_Y[8001:10000]


print ("Train X shape is :", train_X.shape)
print ("Train Y shape is :", train_Y.shape)
print ("Test X shape is :", test_X.shape)
print ("Test Y shape is :", test_Y.shape)

## Normalizing the Input Features: Using tensorflow normalizing function

# Before normalizing, the array of input features should be converted to a dataframe
semitrain_X = pd.DataFrame(train_X)
semitest_X = pd.DataFrame(test_X)
#Importing Scikit learn libraries
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Normalizing Train Data Features
scaler_traindata = scaler.fit(semitrain_X)
train_norm = scaler_traindata.transform(semitrain_X)
X_train=train_norm_X = pd.DataFrame(train_norm)

#Normalizing Test Data Features
scaler_testdata = scaler.fit(semitest_X)
test_norm = scaler_testdata.transform(semitest_X)
X_test=test_norm_X = pd.DataFrame(test_norm)
#Testing/Training Data
Y_train=train_Y
Y_test=test_Y
# Useful paprameters for the Autoencoder
 # dimension one one input data
input_dim = X_train.shape[1]
# this is the size of our encoded representations
encoding_dim = 32 
    


#Training
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l1
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model

## For OMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)
encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim-2), activation='tanh')(code)
decoder = Dense(int(encoding_dim), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

## Parameters for Model configuration and saving 
nb_epoch = 100
batch_size = 60
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
                               
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
                                                

autoencoder = load_model('model.h5')
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
                       
                          