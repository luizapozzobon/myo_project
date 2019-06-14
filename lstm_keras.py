import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

print(tf.VERSION)
print(tf.keras.__version__)

columns = ['Gyro 0', 'Gyro 1', 'Gyro 2', 'Orientation 0', 'Orientation 1', 'Orientation 3', 'Sensor 0', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Label', 'Timestamp']

all_df_true = pd.DataFrame(columns=columns)
all_df_false = pd.DataFrame(columns=columns)


def load_data(path, label):
    global all_df_true, all_df_false
    df = pd.read_csv(path)
    if label:
        all_df_true = pd.concat([all_df_true, df])
    else:
        all_df_false = pd.concat([all_df_false, df])

load_data('datasets/myo-movimento-david-2019-06-06 11:39:44-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:39:58-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:40:54-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:41:03-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:41:12-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:41:54-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:42:25-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:42:39-.csv', True)
load_data('datasets/myo-movimento-david-2019-06-06 11:42:56-.csv', True)

load_data('datasets/myo-movimento-david-ruim-2019-06-06 11:44:11-.csv', False)
load_data('datasets/myo-movimento-david-ruim-2019-06-06 11:44:34-.csv', False)
load_data('datasets/myo-movimento-david-ruim-2019-06-06 11:44:56-.csv', False)
load_data('datasets/myo-movimento-david-ruim-2019-06-06 11:45:18-.csv', False)
load_data('datasets/myo-movimento-david-ruim-2019-06-06 11:45:37-.csv', False)
load_data('datasets/myo-movimento-david-ruim-2019-06-06 11:46:06-.csv', False)


print('All Data:')
print(all_df_true.head())
print(all_df_false.head())

dataset_train = dataset.iloc[np.random.permutation(len(dataset))]
dataset_train.reset_index(drop=True)

X_train = []
y_train = []

for i in range(0, dataset_train.shape[0]):
    row = np.array(dataset_train.iloc[i:1+i, 0:64].values)
    X_train.append(np.reshape(row, (64, 1)))
    y_train.append(np.array(dataset_train.iloc[i:1+i, -1:])[0][0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape to one flatten vector
X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1)
X_train = sc.fit_transform(X_train)

# Reshape again after normalization to (-1, 8, 8)
X_train = X_train.reshape((-1, 8, 8))

# Convert to one hot
y_train = np.eye(np.max(y_train) + 1)[y_train]

print("All Data size X and y")
print(X_train.shape)
print(y_train.shape)

# Splitting Train/Test
X_test = X_train[7700:]
y_test = y_train[7700:]
print("Test Data size X and y")
print(X_test.shape)
print(y_test.shape)

X_train = X_train[0:7700]
y_train = y_train[0:7700]
print("Train Data size X and y")
print(X_train.shape)
print(y_train.shape)

# Creating the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

classifier = Sequential()

classifier.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 8)))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units = 50, return_sequences = True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units = 50, return_sequences = True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units = 50))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 64))
classifier.add(Dense(units = 128))

classifier.add(Dense(units = 4, activation="softmax"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')

classifier.fit(X_train, y_train, epochs = 250, batch_size = 32, verbose=2)

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
