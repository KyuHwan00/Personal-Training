# l1 규제 = 손실함수에 가중치의 절댓값을 더한 경우 => 미분 시, alpha * sign(w) 가짐
# l2 규제 = 손실함수에 가중치의 제곱을 더한 경우  => 미분 시,  alpha * w 를 가짐 

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



dataset = load_breast_cancer()
data = dataset.data
target = dataset.target

x_train_all , x_test , y_train_all , y_test = train_test_split(data, target, stratify= target, test_size= 0.2 , random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify= y_train_all, test_size=0.2, random_state=42)

x_mean = np.mean(x_train, axis = 0)
x_std = np.std(x_train, axis = 0)
x_train_scaled = (x_train - x_mean) / x_std
x_val_scaled = (x_val - x_mean)/ x_std

l1_list = [0.0001, 0.001, 0.01]

for l1 in l1_list:
    sgd = SingleLayer(l1 = l1)
    sgd.fit(x_train_scaled, y_train, x_val= x_val_scaled, y_val= y_val)
    
    plt.plot(sgd.losses)
    plt.plot(sgd.val_losses)
    plt.title('Learning Curve : {0}'.format(l1))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'val_loss'])
    plt.ylim(0,0.3)
    plt.show()


    plt.plot(sgd.w, 'bo')
    plt.ylim(-4,4)
    plt.show()
