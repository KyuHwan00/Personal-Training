import numpy as np

class MulticlassNetwork:
    def __init__(self, batch_size= 32, units = 10, learning_rate = 0.1, l1 = 0, l2 = 0):
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.a1 = None
        self.batch_size = batch_size
        self.units = units
        self.losses = []
        self.val_loss = []
        self.l1 = l1
        self.l2 = l2 
        self.lr = learning_rate
  
    def sigmoid(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1+np.exp(-z))

    def softmax(self, z):
        z = np.clip(z, -100, None)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis= 1).reshape(-1,1)

    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        return z2

    def backprop(self, x, err):
        m = len(x)
        grad_w2 = np.dot(self.a1.T , err) / m # shape = (hidden , output)
        grad_b2 = np.sum(err) / m             # shape = (갯수 , output)
        err_to_hidden = np.dot(err, self.w2.T)*self.a1*(1- self.a1) # shape = (갯수, hidden)
        grad_w1 = np.dot(x.T, err_to_hidden) / m # shape = (특성, hidden)
        grad_b1 = np.sum(err_to_hidden, axis = 0) / m  # shape= (갯수, hidden)
        return grad_w1, grad_b1, grad_w2, grad_b2

    def init_weights(self, n_features, n_classes):
        self.w1 = np.random.normal(0,1, (n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.random.normal(0,1, (self.units, n_classes))
        self.b2 = np.zeros(n_classes)
    
    def training(self, x, y):
        m = len(x)
        z = self.forpass(x)
        a = self.softmax(z)
        err = -(y-a)
        grad_w1, grad_b1, grad_w2, grad_b2 = self.backprop(x, err)
        
        grad_w1 += (self.l1*np.sign(self.w1) + self.l2*self.w1) / m
        grad_w2 += (self.l1*np.sign(self.w2) + self.l2*self.w2) / m

        self.w1 -= self.lr*grad_w1
        self.b1 -= self.lr*grad_b1
        self.w2 -= self.lr*grad_w2
        self.b2 -= self.lr*grad_b2
        return a
    def gen_batch(self, x, y):
        m = len(x)
        bins = m // self.batch_size
        if m % self.batch_size:
            bins +=1
        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size*i
            end = self.batch_size*(i+1)
            yield x[start : end] , y[start : end] # Tap 하나 차이 뭐냐... 뭐냐 하...

    def reg_loss(self):
        return self.l1*(np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + self.l2/2*(np.sum(self.w1**2) + np.sum(self.w2**2))

    def predict(self, x):
        z2 = self.forpass(x)
        return np.argmax(z2, axis = 1)
    
    def score(self,x,y):
        return np.mean(self.predict(x) == np.argmax(y, axis = 1))

    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.softmax(z)
        a = np.clip(a,1e-10, 1-1e-10)
        val_loss = np.sum(-y_val*np.log(a))
        self.val_loss.append((val_loss + self.reg_loss()) / len(y_val))
    
    def fit(self, x,y, epochs = 100, x_val = None , y_val = None):
            np.random.seed(42)
            self.init_weights(x.shape[1], y.shape[1])
            for _ in range(epochs):
                loss = 0
                print('.', end = '')
                for x_batch , y_batch in self.gen_batch(x,y):
                    a = self.training(x_batch, y_batch)
                    a = np.clip(a, 1e-10, 1-1e-10)
                    loss += np.sum(-y_batch*np.log(a))
                self.losses.append((loss + self.reg_loss()) / len(x))
                self.update_val_loss(x_val, y_val)


# MNIST 데이터 세트
import tensorflow as tf
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print(x_train_all.shape, y_train_all.shape)

import matplotlib.pyplot as plt

#plt.imshow(x_train_all[0],'gray')
#plt.show()

from sklearn.model_selection import train_test_split

x_train , x_val, y_train, y_val = train_test_split(x_train_all,y_train_all, stratify= y_train_all, test_size= 0.2, random_state= 42)
#print(np.bincount(y_train))
#print(np.bincount(y_val))

# 표준화가 아닌 0 ~ 1 사이의 값으로 조정 / 이미지 데이터는 각 픽셀마다 0 ~ 255 사이의 값을 가짐
x_train = x_train / 255
x_val = x_val / 255

# 현재 사용중인 클래스는 1차원만 취급하기 때문에 2차원인 이미지 데이터를 1차원으로 reshape해줘야한다 / 이미지의 shape이 (28,28)이기 때문에 784사용
x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)

# 라벨 one-hot encoding 처리해주기 / to_categorical() 문자열은 이 함수를 이용해서 one-hot 불가 / 

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)
fc = MulticlassNetwork(units= 100, batch_size= 256)

fc.fit(x_train, y_train_encoded, x_val= x_val, y_val = y_val_encoded, epochs= 40)
plt.plot(fc.losses)
plt.plot(fc.val_loss)
plt.legend(['train', 'val'])
plt.show()
fc.score(x_val, y_val_encoded)




