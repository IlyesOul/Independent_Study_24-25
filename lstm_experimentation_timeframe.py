import statistics

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# OBJECTIVE: TEST LONG-TERM AND SHORT TERM CAPABILITIES AGAINST VANILLA RNN


data = pd.read_csv('data.csv')
data = data.drop(['Timestamp', 'Ticker', 'Volume'], axis=1)
y = data['High']
X = data.drop(['High'], axis=1)


# FIRST EXPERIMENT: TRAIN ON 3 WEEKs, VALIDATE ANOTHER 3 WEEKs, TEST ON OTHER 3 WEEKS
train_week_x = X[0:21]
validate_week_x = X[21:42]
test_week_x = X[42:63]

train_week_x = np.expand_dims(train_week_x, axis=2)
validate_week_x = np.expand_dims(validate_week_x, axis=2)
test_week_x = np.expand_dims(test_week_x, axis=2)

train_week_y = y[0:21]
validate_week_y = y[21:42]
test_week_y = y[42:63]

# Initialize basic RNN and LSTM

# Building RNN structure
input_layer = tf.keras.layers.Input(shape=(train_week_x.shape[1], 1))

hidden_layer_1 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_2 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_3 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_4 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

output_layer = tf.keras.layers.Dense(1)

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
vanilla_rnn = tf.keras.Sequential([
    input_layer,
    hidden_layer_1,
    hidden_layer_2,
    hidden_layer_3,
    hidden_layer_4,
    output_layer
])

# Building LSTM structure
lstm_input_layer = tf.keras.layers.Input(shape=(train_week_x.shape[1], 1))

lstm_hidden_layer_1 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_2 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_3 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_4 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

lstm_output_layer = tf.keras.layers.Dense(1)

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
lstm_nn = tf.keras.Sequential([
    lstm_input_layer,
    lstm_hidden_layer_1,
    lstm_hidden_layer_2,
    lstm_hidden_layer_3,
    lstm_hidden_layer_4,
    lstm_output_layer
])


vanilla_rnn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
lstm_nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

vanilla_rnn.fit(x=train_week_x, y=train_week_y, validation_data=(validate_week_x, validate_week_y), epochs=50, verbose='auto')
print("Now fitting LSTM")
lstm_nn.fit(x=train_week_x, y=train_week_y, validation_data=(validate_week_x, validate_week_y), epochs=50, verbose='auto',)

# Result Visualization
rnn_predictions = vanilla_rnn.predict(test_week_x).reshape(63, 1, 1).flatten()
lstm_predictions = lstm_nn.predict(test_week_x).reshape(63, 1, 1).flatten()
# Average out every bunch of 3 to solve bug

i = 0
while i < len(rnn_predictions)-2:
    rnn_predictions[i] = statistics.mean([rnn_predictions[i], rnn_predictions[i+1], rnn_predictions[i+2]])
    rnn_predictions[i+1] = 0
    rnn_predictions[i+2] = 0
    i += 3
rnn_predictions = [
    rnn_predictions[i] for i in range(len(rnn_predictions)) if rnn_predictions[i] != 0
]

i = 0
while i < len(lstm_predictions)-2:
    lstm_predictions[i] = statistics.mean([lstm_predictions[i], lstm_predictions[i+1], lstm_predictions[i+2]])
    lstm_predictions[i+1] = 0
    lstm_predictions[i+2] = 0
    i += 3
lstm_predictions = [
    lstm_predictions[i] for i in range(len(lstm_predictions)) if lstm_predictions[i] != 0
]
print(f"Short Term RNN MAE: {mean_absolute_error(test_week_y, rnn_predictions)}\n Short Term LSTM MAE: {mean_absolute_error(test_week_y, lstm_predictions)}")

# Visualization
dates = range(1, 22)
plt.plot(dates, test_week_y, label="True Values")
plt.plot(dates, rnn_predictions, label="RNN Predictions")
plt.plot(dates, lstm_predictions, label="LSTM Predictions")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("RNN vs LSTM Short Term")
plt.legend()
plt.show()


# SECOND EXPERIMENT: TRAIN ON LOTS OF YEARS, TEST ON LOTS YEARS (LONG TERM)
train_week_x = X[0:int(len(X) * .3)]
validate_week_x = X[int(len(X) * .3):int(len(X) * .4)]
test_week_x = X[int(len(X) * .4):int(len(X) * .6)]

train_week_x = np.expand_dims(train_week_x, axis=2)
validate_week_x = np.expand_dims(validate_week_x, axis=2)
test_week_x = np.expand_dims(test_week_x, axis=2)

train_week_y = y[0:int(len(X) * .3)]
validate_week_y = y[int(len(X) * .3):int(len(X) * .4)]
test_week_y = y[int(len(X) * .4):int(len(X) * .6):]

# Initialize basic RNN and LSTM

# Building RNN structure
input_layer = tf.keras.layers.Input(shape=(train_week_x.shape[1], 1))

hidden_layer_1 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_2 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_3 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_4 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

output_layer = tf.keras.layers.Dense(1)

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
vanilla_rnn = tf.keras.Sequential([
    input_layer,
    hidden_layer_1,
    hidden_layer_2,
    hidden_layer_3,
    hidden_layer_4,
    output_layer
])

# Building LSTM structure
lstm_input_layer = tf.keras.layers.Input(shape=(train_week_x.shape[1], 1))

lstm_hidden_layer_1 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_2 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_3 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_4 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

lstm_output_layer = tf.keras.layers.Dense(1)

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
lstm_nn = tf.keras.Sequential([
    lstm_input_layer,
    lstm_hidden_layer_1,
    lstm_hidden_layer_2,
    lstm_hidden_layer_3,
    lstm_hidden_layer_4,
    lstm_output_layer
])


vanilla_rnn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
lstm_nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

vanilla_rnn.fit(x=train_week_x, y=train_week_y, validation_data=(validate_week_x, validate_week_y), epochs=50, verbose='auto')
print("Now fitting LSTM")
lstm_nn.fit(x=train_week_x, y=train_week_y, validation_data=(validate_week_x, validate_week_y), epochs=50, verbose='auto',)

# Result Visualization
rnn_predictions = vanilla_rnn.predict(test_week_x).reshape(1-int(len(X) * .8), 1, 1).flatten()
lstm_predictions = lstm_nn.predict(test_week_x).reshape(1-int(len(X) * .8), 1, 1).flatten()
# Average out every bunch of 3 to solve bug

i = 0
while i < len(rnn_predictions)-2:
    rnn_predictions[i] = statistics.mean([rnn_predictions[i], rnn_predictions[i+1], rnn_predictions[i+2]])
    rnn_predictions[i+1] = 0
    rnn_predictions[i+2] = 0
    i += 3
rnn_predictions = [
    rnn_predictions[i] for i in range(len(rnn_predictions)) if rnn_predictions[i] != 0
]

i = 0
while i < len(lstm_predictions)-2:
    lstm_predictions[i] = statistics.mean([lstm_predictions[i], lstm_predictions[i+1], lstm_predictions[i+2]])
    lstm_predictions[i+1] = 0
    lstm_predictions[i+2] = 0
    i += 3
lstm_predictions = [
    lstm_predictions[i] for i in range(len(lstm_predictions)) if lstm_predictions[i] != 0
]
print(f"Long Term RNN MAE: {mean_absolute_error(test_week_y, rnn_predictions)}\nLong Term LSTM MAE: {mean_absolute_error(test_week_y, lstm_predictions)}")

# Visualization
dates = range(1, len(test_week_x)+1)
plt.plot(dates, test_week_y, label="True Values")
plt.plot(dates, rnn_predictions, label="RNN Predictions")
plt.plot(dates, lstm_predictions, label="LSTM Predictions")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("RNN vs LSTM Long Term")
plt.legend()
plt.show()

# THIRD EXPERIMENT: TRAIN ON MONTHS, TEST ON MONTHS (MID-TERM)
input_layer = tf.keras.layers.Input(shape=(train_week_x.shape[1], 1))

hidden_layer_1 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_2 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_3 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
hidden_layer_4 = tf.keras.layers.SimpleRNN(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

output_layer = tf.keras.layers.Dense(1)

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
vanilla_rnn = tf.keras.Sequential([
    input_layer,
    hidden_layer_1,
    hidden_layer_2,
    hidden_layer_3,
    hidden_layer_4,
    output_layer
])

# Building LSTM structure
lstm_input_layer = tf.keras.layers.Input(shape=(train_week_x.shape[1], 1))

lstm_hidden_layer_1 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_2 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_3 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
lstm_hidden_layer_4 = tf.keras.layers.LSTM(units=500, activation="relu",
                                                   kernel_initializer="glorot_normal", return_sequences=True)
# concatenate_layer = tensorflow.keras.layers.concatenate([hidden_layer_4])

lstm_output_layer = tf.keras.layers.Dense(1)

# recurrent_network = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
lstm_nn = tf.keras.Sequential([
    lstm_input_layer,
    lstm_hidden_layer_1,
    lstm_hidden_layer_2,
    lstm_hidden_layer_3,
    lstm_hidden_layer_4,
    lstm_output_layer
])


vanilla_rnn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
lstm_nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

vanilla_rnn.fit(x=train_week_x, y=train_week_y, validation_data=(validate_week_x, validate_week_y), epochs=50, verbose='auto')
print("Now fitting LSTM")
lstm_nn.fit(x=train_week_x, y=train_week_y, validation_data=(validate_week_x, validate_week_y), epochs=50, verbose='auto',)

# Result Visualization
rnn_predictions = vanilla_rnn.predict(test_week_x).reshape(1-int(len(X) * .8), 1, 1).flatten()
lstm_predictions = lstm_nn.predict(test_week_x).reshape(1-int(len(X) * .8), 1, 1).flatten()
# Average out every bunch of 3 to solve bug

i = 0
while i < len(rnn_predictions)-2:
    rnn_predictions[i] = statistics.mean([rnn_predictions[i], rnn_predictions[i+1], rnn_predictions[i+2]])
    rnn_predictions[i+1] = 0
    rnn_predictions[i+2] = 0
    i += 3
rnn_predictions = [
    rnn_predictions[i] for i in range(len(rnn_predictions)) if rnn_predictions[i] != 0
]

i = 0
while i < len(lstm_predictions)-2:
    lstm_predictions[i] = statistics.mean([lstm_predictions[i], lstm_predictions[i+1], lstm_predictions[i+2]])
    lstm_predictions[i+1] = 0
    lstm_predictions[i+2] = 0
    i += 3
lstm_predictions = [
    lstm_predictions[i] for i in range(len(lstm_predictions)) if lstm_predictions[i] != 0
]
print(f"Mid Term RNN MAE: {mean_absolute_error(test_week_y, rnn_predictions)}\nMid Term LSTM MAE: {mean_absolute_error(test_week_y, lstm_predictions)}")

# Visualization
dates = range(1, len(test_week_x)+1)
plt.plot(dates, test_week_y, label="True Values")
plt.plot(dates, rnn_predictions, label="RNN Predictions")
plt.plot(dates, lstm_predictions, label="LSTM Predictions")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("RNN vs LSTM Mid Term")
plt.legend()
plt.show()
