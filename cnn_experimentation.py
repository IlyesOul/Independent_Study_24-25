import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam


# Loading data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

# Train-test split + scaling
x_train = train_x[:int(.7*len(train_x))]/255.0
validate_x = train_x[int(.7*len(train_x)):]/255.0
test_x = test_x/255.0

y_train = train_y[:int(.7*len(train_y))]
validate_y = train_y[int(.7*len(train_y)):]

# One-hot encoding
train_y = utils.to_categorical(train_y)
validate_y = utils.to_categorical(validate_y)
test_y = utils.to_categorical(test_y)

# Visualize data for self
# for image in train_x:
#     plt.imshow(image)
#     plt.show()

num_filters = 16  # Number of filters (proportional to how many feature maps are created)
kernel_size = (2, 2)  # Size of kernel/filter
pooling_size = (5, 5)  # Max pooling size

cnn = models.Sequential()

# Experiment one: number of filters
cnn.add(tf.keras.layers.Conv2D(filters=16, strides=(1, 1), kernel_size=kernel_size))  # 16 Filters
# cnn.add(tf.keras.layers.Conv2D(filters=16, strides=(1, 1), kernel_size=kernel_size))  # 32 Filters
# cnn.add(tf.keras.layers.Conv2D(filters=16, strides=(1, 1), kernel_size=kernel_size))  # 8 Filters
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=pooling_size))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=150, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=150, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=10, activation="softmax"))

adam = Adam(learning_rate=.001)
cnn.compile(optimizer=adam, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
kernel_history = cnn.fit(x=train_x, y=train_y, validation_data=(validate_x, validate_y), verbose='auto', epochs=50)
print(f"Prediction loss: {tf.keras.losses.categorical_crossentropy(test_y, cnn.predict(test_x))}")

#callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
