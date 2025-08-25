import math
import matplotlib.pyplot as mlp
import tensorflow.keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_absolute_error

# DATA PREPROCESSING AND CLEANING
data = pd.read_csv('data.csv')
data = data.drop(['Timestamp', 'Ticker', 'Volume'], axis=1)
y = data['High']
X = data.drop(['High'], axis=1)
len_data = len(data)
print(f"x: {X}")
print(f"y: {y}")

# Train, test, validation split
training_x = X[0:math.floor(len_data * .6)]
training_y = y[0:math.floor(len_data * .6)]

validation_x = X[math.floor(len_data * .6):math.floor(len_data * .8)]
validation_y = y[math.floor(len_data * .6):math.floor(len_data * .8)]

test_x = X[math.floor(len_data * .8):]
test_y = y[math.floor(len_data * .8):]

# MinMax Normalization
sc = MinMaxScaler()

# training_x = sc.fit_transform(training_x)
# validation_x = sc.fit_transform(validation_x)
# test_x = sc.fit_transform(test_x)

# Reshaping Data
# training_x = np.expand_dims(training_x, axis=2)
# validation_x = np.expand_dims(validation_x, axis=2)
# test_x = np.expand_dims(test_x, axis=2)
# END OF DATA PREPROCESSING AND CLEANING

# STOCK PREDICTIONS WITH RNN

# Building RNN structure
input_layer = tensorflow.keras.layers.Input(shape=(training_x.shape[1], 1))
print(training_x)
hidden_layer_1 = tensorflow.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_2 = tensorflow.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_3 = tensorflow.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_4 = tensorflow.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

output_layer = tensorflow.keras.layers.Dense(1, activation="relu")

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
recurrent_network = tensorflow.keras.Sequential([
    input_layer,
    hidden_layer_1,
    hidden_layer_2,
    hidden_layer_3,
    hidden_layer_4,
    output_layer
])

# Training and compiling network
recurrent_network.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error", "accuracy"])
recurrent_network.fit(training_x, training_y, epochs=30, verbose='auto', callbacks=[tensorflow.keras.callbacks.EarlyStopping(patience=3)])

# Visualize backpropagation
model_history = recurrent_network.history()
