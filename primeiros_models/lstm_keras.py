import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

print(tf.VERSION)
print(tf.keras.__version__)

#columns = ['Gyro 0', 'Gyro 1', 'Gyro 2', 'Orientation 0', 'Orientation 1', 'Orientation 3', 'Sensor 0', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Label', 'Timestamp']

columns = ['Gyro 0', 'Gyro 1', 'Gyro 2', 'Orientation 0', 'Orientation 1', 'Orientation 3', 'Sensor 0', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Label']

x_train = pd.DataFrame(columns=columns)
x_test = pd.DataFrame(columns=columns)


def load_data():
    global x_train, x_test
    path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"
    path_bom = path + 'mov_bom'
    path_ruim = path + 'mov_ruim'
    files_bom = glob.glob(path_bom+"/*.csv")
    files_ruim = glob.glob(path_ruim+"/*.csv")
    train_test_split_bom = round(len(files_bom)*0.7)
    train_test_split_ruim = round(len(files_ruim)*0.7)
    count = 0
    for f in files_bom:
        df = pd.read_csv(f)
        df = df.drop(columns=["Timestamp", "Unnamed: 0"])
        if count <= train_test_split_bom:
            x_train = pd.concat([x_train, df], sort=False)
        else:
            x_test = pd.concat([x_test, df], sort=False)
        count += 1
    count = 0
    for f in files_ruim:
        df = pd.read_csv(f)
        df = df.drop(columns=["Timestamp", "Unnamed: 0"])
        if count <= train_test_split_ruim:
            x_train = pd.concat([x_train, df], sort=False)
        else:
            x_test = pd.concat([x_test, df], sort=False)
        count += 1

def get_labels():
    global y_test, y_train, x_train, x_test
    """Get labels columns from df and drop them"""
    y_train = x_train["Label"]
    x_train = x_train.drop(columns=["Label"])
    y_test = x_test["Label"]
    x_test = x_test.drop(columns=["Label"])

load_data()
get_labels()


print('All Data:')
print(x_train.head(), x_train.shape)
print(x_test.head(), x_test.shape)
print(y_train)
print(y_test)

print(x_train.shape)

X_train = np.array(x_train)
y_train = np.array(y_train)

# batch_input_shape=[z, x, y]
#   -> z = batch_size (nº de amostras)
#   -> x = time steps (quantas leituras)
#   -> y = input_units
# https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
# samples, timesteps, features

X_train = X_train.reshape(43, 500, 15)
print(X_train.shape)

y_train = y_train.reshape(43, 1, 500)
new_y_train = list()
for i, label in enumerate(y_train):
    if 0 in label:
        new_y_train.append(0)
    elif 1 in label:
        new_y_train.append(1)

print(X_train)
print(y_train)
print(new_y_train)

y_train = np.array(new_y_train)

# Creating the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

input_shape = (X_train.shape[1], X_train.shape[2])
print(input_shape, y_train.shape, X_train.shape)

# LER ISSO
# https://stackoverflow.com/questions/53382353/lstm-input-shape-for-multivariate-time-series

optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

classifier = Sequential()

classifier.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
classifier.add(Dropout(0.1))

classifier.add(LSTM(units = 50, return_sequences = True))
classifier.add(Dropout(0.1))

classifier.add(LSTM(units = 50, return_sequences = True))
classifier.add(Dropout(0.1))

classifier.add(LSTM(units = 50))
classifier.add(Dropout(0.1))

classifier.add(Dense(units = 64))
classifier.add(Dense(units = 128))

classifier.add(Dense(units = 1, activation="softmax"))

classifier.summary()

classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=["accuracy"])

classifier.fit(X_train, y_train, epochs = 250, batch_size = 43, verbose=2)

# Save
classifier.save("model_cross_splited_data.h5")
print("Saved model to disk")

###############################################

from tensorflow import keras

# # Load Model
# model = keras.models.load_model('model_cross_splited_data.h5')
# model.summary()

def evaluateModel(prediction, y):
    good = 0
    for i in range(len(y)):
        if (prediction[i] == np.argmax(y[i])):
            good = good +1
    return (good/len(y)) * 100.0

result_test = classifier.predict_classes(X_test)
print("Correct classification rate on test data")
print(evaluateModel(result_test, y_test))

result_train = classifier.predict_classes(X_train)
print("Correct classification rate on train data")
print(evaluateModel(result_train, y_train))
