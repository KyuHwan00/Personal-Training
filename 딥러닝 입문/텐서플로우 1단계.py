import tensorflow as tf


(x_train_all, y_train_all) , (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train_all.shape, y_train_all.shape)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify= y_train_all, random_state=42, test_size= 0.2)

x_train = x_train / 255
x_val = x_val / 255

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)

x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()

model.add(Dense(units= 100, activation= 'sigmoid', input_shape = (784,)))
model.add(Dense(units= 10, activation = 'softmax'))

model.compile(optimizer= 'sgd', loss = 'categorical_crossentropy', metrics= ['accuracy'])

history = model.fit(x_train, y_train_encoded, epochs= 40, validation_data= (x_val, y_val_encoded))

print(history.history.keys())
       