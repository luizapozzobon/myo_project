from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def load_compilado(arquivo):
    path = '/home/luiza/UFSM/Myo/myo_project/datasets/oficial/' + arquivo
    df = pd.read_csv(path)
    return df

df = load_compilado('features_balanced.csv')
df_labels = df['Label']
df = df.drop(columns=['Label'])

x_train, x_test, y_train, y_test = train_test_split(df.values, df_labels.values, test_size=0.3, random_state=0)

print('All Data:')
print(y_train)
print(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam

input_shape = (x_train.shape[1])
print(input_shape, y_train.shape, x_train.shape)

optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

classifier = Sequential()
classifier.add(Dense(32, input_dim=x_train.shape[1]))
classifier.add(Activation('relu'))
classifier.add(Dense(units = 64))
classifier.add(Activation('relu'))
classifier.add(Dense(units = 128))
classifier.add(Activation('relu'))
classifier.add(Dense(units = 1, activation="softmax"))

classifier.summary()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])

classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=["accuracy"])

classifier.fit(x_train, y_train, epochs = 10000, batch_size = 43, verbose=1)

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

# result_test = classifier.predict_classes(X_test)
# print("Correct classification rate on test data")
# print(evaluateModel(result_test, y_test))

result_train = classifier.predict_classes(x_train)
print("Correct classification rate on train data")
print(evaluateModel(result_train, y_train))
