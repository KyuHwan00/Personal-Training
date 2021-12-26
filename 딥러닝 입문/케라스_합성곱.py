import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
import tensorflow as tf
from sklearn.model_selection import train_test_split



(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify= y_train_all, test_size= 0.2, random_state= 42)

x_train = x_train[...,tf.newaxis] / 255
x_val = x_val.reshape(-1,28,28,1) / 255
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)




conv1 = tf.keras.Sequential()



conv1.add(Conv2D(10, (3,3), strides= 1, padding= 'SAME', activation= 'relu', input_shape= (28,28,1)))


conv1.add(MaxPool2D((2,2)))
conv1.add(Flatten())
conv1.add(Dense(100, activation= 'relu'))
conv1.add(Dense(10, activation= 'softmax'))

conv1.summary()

conv1.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])

history = conv1.fit(x_train, y_train_encoded, epochs = 20, validation_data = (x_val, y_train_encoded))


