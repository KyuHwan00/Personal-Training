from numpy import load
from sklearn.datasets import load_breast_cancer

cancer  = load_breast_cancer() # Bunch 클래스

x = cancer.data
y = cancer.target

print(x.shape, y.shape)

# 여러 특성 열거한 그래프

import matplotlib.pyplot as plt
import numpy as np

plt.boxplot(x)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
plt.close()

np.unique(y, return_counts = True)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, stratify= y, test_size= 0.2 ,random_state=42)

class LogisticNeuron:
    def __init__(self):
        self.w = None
        self.b = None
    def forpass(self, x):
        y_hat = np.sum(self.w*x) + self.b
        return y_hat

    def backprop(self, x, y):
        grad_w = -(y-self.activation(self.forpass(x)))*x
        grad_b = -(y-self.activation(self.forpass(x)))
        return grad_w, grad_b

    def activation(self,x):
        return 1/(1+np.exp(-x))

    def fit(self, x, y):
        self.w = np.ones(x.shape[1])
        self.b = 1
        for _ in range(100):
            for x_i, y_i in zip(x,y):
                grad_w, grad_b = self.backprop(x_i,y_i)
                self.w -= grad_w
                self.b -= grad_b
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a >0.5

neuron  = LogisticNeuron()
neuron.fit(x_train, y_train)

np.mean(neuron.predict(x_test) == y_test)