import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Loading data
(x_train, x_test), (y_train, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"Shape of x_train: {(x_train).shape}")
