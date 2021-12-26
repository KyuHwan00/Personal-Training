from numpy.lib.function_base import median
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class SingleLayer():
    def __init__(self, learning_rate =0.1, l1 = 0 , l2 = 0):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.l1 = l1
        self.l2 = l2
    def forpass(self, x):
        return np.sum(self.w*x) + self.b
    
    def activation(self, z):
        return 1/(1+np.exp(-z))

    def backprop(self, x , err):
        grad_w = -err*x
        grad_b = -err
        return grad_w, grad_b
    def reg_loss(self):
        return self.l1*np.sum(np.abs(self.w)) + self.l2*np.sum(self.w** 2)/2
    
    def fit(self, x, y, epochs = 100, x_val = None , y_val = None):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for _ in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = y[i] - a
                grad_w , grad_b = self.backprop(x[i], err)
                grad_w += self.l1*np.sign(self.w) + self.l2*self.w
                self.w -= self.lr*grad_w
                self.b -= grad_b
                self.w_history.append(self.w.copy())
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a) + (1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y) + self.reg_loss())
            self.update_val_loss(x_val, y_val)
    def predict(self, x_t):
        z = [self.forpass(x) for x in x_t]
        return np.array(z) > 0 
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            loss += -(y_val[i]*np.log(a) + (1-y_val[i])*np.log(1-a))
        self.val_losses.append(loss/len(y_val)+ self.reg_loss())


validation_scores = []
dataset = load_breast_cancer()
data = dataset.data
target = dataset.target

x_train_all , x_test , y_train_all , y_test = train_test_split(data, target, stratify= target, test_size= 0.2 , random_state=42)

k = 10
bins = len(x_train_all)//k

for i in range(k):
    start = i*bins
    end = (i+1)*bins

    x_val = x_train_all[start:end]
    y_val = y_train_all[start:end]
    index = list(range(0, start)) + list(range(end, len(x_train_all)))
    x_train = x_train_all[index]
    y_train = y_train_all[index]

    mean = np.mean(x_train, axis = 0)
    std = np.std(x_train, axis = 0)
    x_train_norm  = (x_train - mean)/std
    x_val_norm = (x_val - mean)/std

    lyr = SingleLayer(l2 = 0.01)
    lyr.fit(x_train_norm, y_train)
    score = lyr.score(x_val_norm, y_val)
    validation_scores.append(score)
print(np.mean(validation_scores))